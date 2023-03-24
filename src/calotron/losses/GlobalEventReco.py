import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy as TF_BCE
from tensorflow.keras.losses import MeanSquaredError as TF_MSE

from calotron.losses.BaseLoss import BaseLoss


class GlobalEventReco(BaseLoss):
    def __init__(
        self,
        alpha=0.1,
        noise_stddev=0.05,
        from_logits=False,
        label_smoothing=0.0,
        name="event_reco_loss",
    ) -> None:
        super().__init__(name)

        # Adversarial strength
        assert isinstance(alpha, (int, float))
        assert alpha >= 0.0
        self._alpha = float(alpha)

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

        # TensorFlow MeanSquaredError
        self._mse_loss = TF_MSE(reduction="auto")

        # TensorFlow BinaryCrossentropy
        self._bce_loss = TF_BCE(
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
        # Global event reco loss
        output = transformer((source, target), training=training)
        mse_loss = self._mse_loss(target, output, sample_weight=sample_weight)
        mse_loss = tf.cast(mse_loss, dtype=target.dtype)

        # Adversarial loss
        rnd_pred = tf.random.normal(
            tf.shape(output), stddev=self._noise_stddev, dtype=output.dtype
        )
        y_pred = discriminator(output + rnd_pred, training=False)
        adv_loss = self._bce_loss(
            tf.ones_like(y_pred), y_pred, sample_weight=sample_weight
        )
        adv_loss = tf.cast(adv_loss, dtype=output.dtype)
        return mse_loss + self._alpha * adv_loss

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
        real_loss = self._bce_loss(
            tf.ones_like(y_true), y_true, sample_weight=sample_weight
        )
        real_loss = tf.cast(real_loss, dtype=target.dtype)

        # Fake target loss
        rnd_pred = tf.random.normal(
            tf.shape(output), stddev=self._noise_stddev, dtype=output.dtype
        )
        y_pred = discriminator(output + rnd_pred, training=training)
        fake_loss = self._bce_loss(
            tf.zeros_like(y_pred), y_pred, sample_weight=sample_weight
        )
        fake_loss = tf.cast(fake_loss, dtype=output.dtype)
        return (real_loss + fake_loss) / 2.0

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def noise_stddev(self) -> float:
        return self._noise_stddev

    @property
    def from_logits(self) -> bool:
        return self._from_logits

    @property
    def label_smoothing(self) -> float:
        return self._label_smoothing
