import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy as TF_BCE

from calotron.losses.BaseLoss import BaseLoss


class BinaryCrossentropy(BaseLoss):
    def __init__(
        self,
        noise_stddev=0.05,
        from_logits=False,
        label_smoothing=0.0,
        name="bce_loss",
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

    def discriminator_loss(
        self,
        discriminator,
        source_true,
        target_true,
        target_pred,
        sample_weight=None,
        discriminator_training=True,
    ) -> tf.Tensor:
        # Real target loss
        rnd_true = tf.random.normal(
            tf.shape(target_true), stddev=self._noise_stddev, dtype=target_true.dtype
        )
        y_true = discriminator(target_true + rnd_true, training=discriminator_training)
        real_loss = self._loss(
            tf.ones_like(y_true), y_true, sample_weight=sample_weight
        )
        real_loss = tf.cast(real_loss, dtype=target_true.dtype)

        # Fake target loss
        rnd_pred = tf.random.normal(
            tf.shape(target_pred), stddev=self._noise_stddev, dtype=target_pred.dtype
        )
        y_pred = discriminator(target_pred + rnd_pred, training=discriminator_training)
        fake_loss = self._loss(
            tf.zeros_like(y_pred), y_pred, sample_weight=sample_weight
        )
        fake_loss = tf.cast(fake_loss, dtype=target_pred.dtype)
        return (real_loss + fake_loss) / 2.0

    def transformer_loss(
        self,
        discriminator,
        source_true,
        target_true,
        target_pred,
        sample_weight=None,
        discriminator_training=False,
    ) -> tf.Tensor:
        # Adversarial loss
        rnd_pred = tf.random.normal(
            tf.shape(target_pred), stddev=self._noise_stddev, dtype=target_pred.dtype
        )
        y_pred = discriminator(target_pred + rnd_pred, training=discriminator_training)
        adv_loss = self._loss(
            tf.ones_like(y_pred), y_pred, sample_weight=sample_weight
        )
        adv_loss = tf.cast(adv_loss, dtype=target_pred.dtype)
        return adv_loss
    
    @property
    def noise_stddev(self) -> float:
        return self._noise_stddev
    
    @property
    def from_logits(self) -> bool:
        return self._from_logits
    
    @property
    def label_smoothing(self) -> float:
        return self._label_smoothing
