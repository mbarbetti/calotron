import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy as TF_BCE

from calotron.losses.BaseLoss import BaseLoss


class BinaryCrossentropy(BaseLoss):
    def __init__(
        self, noise_stddev=0.05, from_logits=False, label_smoothing=0.0, name="bce_loss"
    ) -> None:
        super().__init__(name)

        # Noise standard deviation
        assert isinstance(noise_stddev, (int, float))
        assert noise_stddev > 0.0
        self._noise_stddev = noise_stddev

        # BCE `from_logits` flag
        assert isinstance(from_logits, bool)
        self._from_logits = from_logits

        # BCE `label_smoothing`
        assert isinstance(label_smoothing, (int, float))
        assert label_smoothing >= 0.0 and label_smoothing <= 1.0
        self._label_smoothing = label_smoothing

        # TensorFlow BinaryCrossentropy
        self._loss = TF_BCE(
            from_logits=self._from_logits,
            label_smoothing=self._label_smoothing,
            axis=-1,
            reduction="auto",
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
        # Adversarial loss
        output = transformer((source, target), training=training)
        rnd_pred = tf.random.normal(
            tf.shape(output), stddev=self._noise_stddev, dtype=output.dtype
        )
        y_pred = discriminator(output + rnd_pred, training=False)
        adv_loss = self._loss(tf.ones_like(y_pred), y_pred, sample_weight=sample_weight)
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
        # Real target loss
        output = transformer((source, target), training=False)
        rnd_true = tf.random.normal(
            tf.shape(target), stddev=self._noise_stddev, dtype=target.dtype
        )
        y_true = discriminator(target + rnd_true, training=training)
        real_loss = self._loss(
            tf.ones_like(y_true), y_true, sample_weight=sample_weight
        )
        real_loss = tf.cast(real_loss, dtype=target.dtype)

        # Fake target loss
        rnd_pred = tf.random.normal(
            tf.shape(output), stddev=self._noise_stddev, dtype=output.dtype
        )
        y_pred = discriminator(output + rnd_pred, training=training)
        fake_loss = self._loss(
            tf.zeros_like(y_pred), y_pred, sample_weight=sample_weight
        )
        fake_loss = tf.cast(fake_loss, dtype=output.dtype)
        return (real_loss + fake_loss) / 2.0

    @property
    def noise_stddev(self) -> float:
        return self._noise_stddev

    @property
    def from_logits(self) -> bool:
        return self._from_logits

    @property
    def label_smoothing(self) -> float:
        return self._label_smoothing
