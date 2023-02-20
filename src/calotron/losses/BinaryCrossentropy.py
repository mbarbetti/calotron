import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy as TF_BCE

from calotron.losses.BaseLoss import BaseLoss


class BinaryCrossentropy(BaseLoss):
    def __init__(
        self,
        from_logits=False,
        label_smoothing=0.0,
        axis=-1,
        reduction="auto",
        name="bce_loss",
    ) -> None:
        super().__init__(name)
        self._loss = TF_BCE(
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            axis=axis,
            reduction=reduction,
        )

    def discriminator_loss(
        self, discriminator, target_true, target_pred, sample_weight=None
    ) -> tf.Tensor:
        rnd_true = tf.random.normal(
            tf.shape(target_true), stddev=0.05, dtype=target_true.dtype
        )
        y_true = discriminator(target_true + rnd_true, training=True)
        loss_real = self._loss(
            tf.ones_like(y_true), y_true, sample_weight=sample_weight
        )
        loss_real = tf.cast(loss_real, dtype=target_true.dtype)
        rnd_pred = tf.random.normal(
            tf.shape(target_pred), stddev=0.05, dtype=target_pred.dtype
        )
        y_pred = discriminator(target_pred + rnd_pred, training=True)
        loss_fake = self._loss(
            tf.zeros_like(y_pred), y_pred, sample_weight=sample_weight
        )
        loss_fake = tf.cast(loss_fake, dtype=target_pred.dtype)
        return (loss_real + loss_fake) / 2.0

    def transformer_loss(
        self, discriminator, target_true, target_pred, sample_weight=None
    ) -> tf.Tensor:
        rnd_pred = tf.random.normal(
            tf.shape(target_pred), stddev=0.05, dtype=target_pred.dtype
        )
        y_pred = discriminator(target_pred + rnd_pred, training=False)
        loss_fake = self._loss(
            tf.ones_like(y_pred), y_pred, sample_weight=sample_weight
        )
        loss_fake = tf.cast(loss_fake, dtype=target_pred.dtype)
        return loss_fake
