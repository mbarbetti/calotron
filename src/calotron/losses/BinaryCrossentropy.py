import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy as TF_BCE

from calotron.losses.BaseLoss import BaseLoss


class BinaryCrossentropy(BaseLoss):
    def __init__(
        self,
        injected_noise_stddev=0.0,
        from_logits=False,
        label_smoothing=0.0,
        warmup_energy=0.0,
        name="bce_loss",
    ) -> None:
        super().__init__(name)

        # Warmup energy
        assert isinstance(warmup_energy, (int, float))
        assert warmup_energy >= 0.0
        self._warmup_energy = float(warmup_energy)

        # Noise standard deviation
        assert isinstance(injected_noise_stddev, (int, float))
        assert injected_noise_stddev >= 0.0
        self._inj_noise_std = float(injected_noise_stddev)

        # BCE `from_logits` flag
        assert isinstance(from_logits, bool)
        self._from_logits = from_logits

        # BCE `label_smoothing`
        assert isinstance(label_smoothing, (int, float))
        assert label_smoothing >= 0.0 and label_smoothing <= 1.0
        self._label_smoothing = float(label_smoothing)

        # TensorFlow BinaryCrossentropy
        self._loss = TF_BCE(
            from_logits=self._from_logits, label_smoothing=self._label_smoothing
        )

    def transformer_loss(
        self,
        transformer,
        discriminator,
        source,
        target,
        sample_weight=None,
        training=True,
    ) -> tf.Tensor:
        _, trainset_pred = self._prepare_adv_trainset(
            source=source,
            target=target,
            transformer=transformer,
            warmup_energy=self._warmup_energy,
            sample_weight=sample_weight,
            training_transformer=training,
        )
        source_pred, target_pred, evt_w_pred, mask_pred = trainset_pred

        if self._inj_noise_std > 0.0:
            rnd_pred = tf.random.normal(
                tf.shape(target_pred),
                stddev=self._inj_noise_std,
                dtype=target_pred.dtype,
            )
        else:
            rnd_pred = 0.0

        y_pred = discriminator(
            (source_pred, target_pred + rnd_pred),
            padding_mask=mask_pred,
            training=False,
        )

        # Adversarial loss
        adv_loss = self._loss(tf.ones_like(y_pred), y_pred, sample_weight=evt_w_pred)
        adv_loss = tf.cast(adv_loss, dtype=target_pred.dtype)
        return adv_loss

    def discriminator_loss(
        self,
        transformer,
        discriminator,
        source,
        target,
        sample_weight=None,
        training=True,
    ) -> tf.Tensor:
        trainset_true, trainset_pred = self._prepare_adv_trainset(
            source=source,
            target=target,
            transformer=transformer,
            warmup_energy=self._warmup_energy,
            sample_weight=sample_weight,
            training_transformer=False,
        )
        source_true, target_true, evt_w_true, mask_true = trainset_true
        source_pred, target_pred, evt_w_pred, mask_pred = trainset_pred

        if self._inj_noise_std > 0.0:
            rnd_noise = tf.random.normal(
                shape=(
                    2 * tf.shape(target_true)[0],
                    tf.shape(target_true)[1],
                    tf.shape(target_true)[2],
                ),
                stddev=self._inj_noise_std,
                dtype=target.dtype,
            )
        else:
            rnd_noise = 0.0

        source_concat = tf.concat([source_true, source_pred], axis=0)
        target_concat = tf.concat([target_true, target_pred], axis=0)
        mask_concat = tf.concat([mask_true, mask_pred], axis=0)
        d_out = discriminator(
            (source_concat, target_concat + rnd_noise),
            padding_mask=mask_concat,
            training=training,
        )
        y_true, y_pred = tf.split(d_out, 2, axis=0)

        # Real target loss
        real_loss = self._loss(tf.ones_like(y_true), y_true, sample_weight=evt_w_true)
        real_loss = tf.cast(real_loss, dtype=target_true.dtype)

        # Fake target loss
        fake_loss = self._loss(tf.zeros_like(y_pred), y_pred, sample_weight=evt_w_pred)
        fake_loss = tf.cast(fake_loss, dtype=target_pred.dtype)

        if discriminator.condition_aware:
            shuffled_source = tf.random.shuffle(source_true)
            masked_target = target_true * tf.tile(
                mask_true[:, :, None], (1, 1, tf.shape(target_true)[2])
            )
            y_pred = discriminator(
                (shuffled_source, masked_target), padding_mask=None, training=training
            )

            # Fake source loss
            source_loss = self._loss(tf.zeros_like(y_pred), y_pred, sample_weight=None)
            source_loss = tf.cast(source_loss, dtype=target_pred.dtype)

            return (real_loss + fake_loss + source_loss) / 3.0
        else:
            return (real_loss + fake_loss) / 2.0

    @property
    def warmup_energy(self) -> float:
        return self._warmup_energy

    @property
    def injected_noise_stddev(self) -> float:
        return self._inj_noise_std

    @property
    def from_logits(self) -> bool:
        return self._from_logits

    @property
    def label_smoothing(self) -> float:
        return self._label_smoothing
