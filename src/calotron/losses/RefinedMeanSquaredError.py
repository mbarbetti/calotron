import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy as TF_BCE
from tensorflow.keras.losses import MeanSquaredError as TF_MSE

from calotron.losses.BaseLoss import BaseLoss

STD_DEV = 0.02


class RefinedMeanSquaredError(BaseLoss):
    def __init__(
        self,
        alpha=0.1,
        discriminator_from_logits=False,
        discriminator_label_smoothing=0.0,
        name="refined_mse_loss",
    ) -> None:
        super().__init__(name)

        # Balance btw MSE and BCE
        assert isinstance(alpha, (int, float))
        assert alpha >= 0.0
        self._alpha = float(alpha)

        # TensorFlow MeanSquaredError
        self._mse_loss = TF_MSE(reduction="auto")

        # TensorFlow BinaryCrossentropy
        self._bce_loss = TF_BCE(
            from_logits=discriminator_from_logits,
            label_smoothing=discriminator_label_smoothing,
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
        # Real loss computation
        rnd_true = tf.random.normal(
            tf.shape(target_true), stddev=STD_DEV, dtype=target_true.dtype
        )
        y_true = discriminator(target_true + rnd_true, training=discriminator_training)
        loss_real = self._bce_loss(
            tf.ones_like(y_true), y_true, sample_weight=sample_weight
        )
        loss_real = tf.cast(loss_real, dtype=target_true.dtype)

        # Fake loss computation
        rnd_pred = tf.random.normal(
            tf.shape(target_pred), stddev=STD_DEV, dtype=target_pred.dtype
        )
        y_pred = discriminator(target_pred + rnd_pred, training=discriminator_training)
        loss_fake = self._bce_loss(
            tf.zeros_like(y_pred), y_pred, sample_weight=sample_weight
        )
        loss_fake = tf.cast(loss_fake, dtype=target_pred.dtype)
        return (loss_real + loss_fake) / 2.0

    def transformer_loss(
        self,
        discriminator,
        source_true,
        target_true,
        target_pred,
        sample_weight=None,
        discriminator_training=False,
    ) -> tf.Tensor:
        # MSE loss computation
        mse_loss = self._mse_loss(target_true, target_pred, sample_weight=sample_weight)
        mse_loss = tf.cast(mse_loss, dtype=target_true.dtype)

        # Adversarial loss computation
        rnd_pred = tf.random.normal(
            tf.shape(target_pred), stddev=STD_DEV, dtype=target_pred.dtype
        )
        y_pred = discriminator(target_pred + rnd_pred, training=discriminator_training)
        adv_loss = self._bce_loss(
            tf.ones_like(y_pred), y_pred, sample_weight=sample_weight
        )
        adv_loss = tf.cast(adv_loss, dtype=target_pred.dtype)
        return mse_loss + self._alpha * adv_loss
