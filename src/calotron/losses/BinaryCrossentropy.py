import tensorflow as tf
from tensorflow import keras

from calotron.losses.BaseLoss import BaseLoss


class BinaryCrossentropy(BaseLoss):
    def __init__(
        self,
        injected_noise_stddev=0.0,
        from_logits=False,
        label_smoothing=0.0,
        warmup_energy=1e-8,
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
        self._loss = keras.losses.BinaryCrossentropy(
            from_logits=self._from_logits,
            label_smoothing=self._label_smoothing,
            reduction=tf.keras.losses.Reduction.NONE,
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
        _, y_pred, evt_weights, _ = self._perform_classification(
            source=source,
            target=target,
            transformer=transformer,
            discriminator=discriminator,
            warmup_energy=self._warmup_energy,
            inj_noise_std=self._inj_noise_std,
            sample_weight=sample_weight,
            training_transformer=training,
            training_discriminator=False,
            return_transformer_output=False,
        )

        # Adversarial loss
        adv_loss = self._loss(tf.ones_like(y_pred), y_pred)
        adv_loss = tf.reduce_sum(evt_weights * adv_loss) / tf.reduce_sum(evt_weights)
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
        y_true, y_pred, evt_weights, mask = self._perform_classification(
            source=source,
            target=target,
            transformer=transformer,
            discriminator=discriminator,
            warmup_energy=self._warmup_energy,
            inj_noise_std=self._inj_noise_std,
            sample_weight=sample_weight,
            training_transformer=training,
            training_discriminator=False,
            return_transformer_output=False,
        )

        # Real target loss
        real_loss = self._loss(tf.ones_like(y_true), y_true)
        real_loss = tf.reduce_sum(evt_weights * real_loss) / tf.reduce_sum(evt_weights)

        # Fake target loss
        fake_loss = self._loss(tf.zeros_like(y_pred), y_pred)
        fake_loss = tf.reduce_sum(evt_weights * fake_loss) / tf.reduce_sum(evt_weights)

        if discriminator.condition_aware:
            source_shuffle = tf.random.shuffle(source)
            y_pred = discriminator(
                (source_shuffle, target), padding_mask=mask, training=training
            )

            # Fake source loss
            source_loss = self._loss(tf.zeros_like(y_pred), y_pred)
            source_loss = tf.reduce_mean(source_loss)

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
