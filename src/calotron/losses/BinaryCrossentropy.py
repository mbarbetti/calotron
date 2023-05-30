import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy as TF_BCE

from calotron.losses.BaseLoss import BaseLoss


class BinaryCrossentropy(BaseLoss):
    def __init__(
        self,
        warmup_energy=0.0,
        injected_noise_stddev=0.0,
        from_logits=False,
        label_smoothing=0.0,
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
        output = transformer((source, target), training=training)
        energy_mask = tf.cast(
            target[:, :, 2] >= self._warmup_energy, dtype=target.dtype
        )
        if sample_weight is None:
            sample_weight = tf.identity(energy_mask)
        else:
            sample_weight *= energy_mask

        # Adversarial loss
        if self._inj_noise_std > 0.0:
            rnd_pred = tf.random.normal(
                tf.shape(output), stddev=self._inj_noise_std, dtype=output.dtype
            )
        else:
            rnd_pred = 0.0
        y_pred = discriminator(
            (source, output + rnd_pred), filter=sample_weight, training=False
        )
        adv_loss = self._loss(tf.ones_like(y_pred), y_pred)
        adv_loss = tf.cast(adv_loss, dtype=output.dtype)
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
        output = transformer((source, target), training=False)
        energy_mask = tf.cast(
            target[:, :, 2] >= self._warmup_energy, dtype=target.dtype
        )
        if sample_weight is None:
            sample_weight = tf.identity(energy_mask)
        else:
            sample_weight *= energy_mask

        # Real target loss
        if self._inj_noise_std > 0.0:
            rnd_true = tf.random.normal(
                tf.shape(target), stddev=self._inj_noise_std, dtype=target.dtype
            )
        else:
            rnd_true = 0.0
        y_true = discriminator(
            (source, target + rnd_true), filter=sample_weight, training=training
        )
        real_loss = self._loss(tf.ones_like(y_true), y_true)
        real_loss = tf.cast(real_loss, dtype=target.dtype)

        # Fake target loss
        if self._inj_noise_std > 0.0:
            rnd_pred = tf.random.normal(
                tf.shape(output), stddev=self._inj_noise_std, dtype=output.dtype
            )
        else:
            rnd_pred = 0.0
        y_pred = discriminator(
            (source, output + rnd_pred), filter=sample_weight, training=training
        )
        fake_loss = self._loss(tf.zeros_like(y_pred), y_pred)
        fake_loss = tf.cast(fake_loss, dtype=output.dtype)
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
