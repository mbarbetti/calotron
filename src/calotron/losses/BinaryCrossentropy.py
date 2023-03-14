import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy as TF_BCE

from calotron.losses.BaseLoss import BaseLoss

STD_DEV = 0.02


class BinaryCrossentropy(BaseLoss):
    def __init__(
        self,
        discriminator_from_logits=False,
        discriminator_label_smoothing=0.0,
        name="bce_loss",
    ) -> None:
        super().__init__(name)
        self._loss = TF_BCE(
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
        loss_real = self._loss(
            tf.ones_like(y_true), y_true, sample_weight=sample_weight
        )
        loss_real = tf.cast(loss_real, dtype=target_true.dtype)

        # Fake loss computation
        rnd_pred = tf.random.normal(
            tf.shape(target_pred), stddev=STD_DEV, dtype=target_pred.dtype
        )
        y_pred = discriminator(target_pred + rnd_pred, training=discriminator_training)
        loss_fake = self._loss(
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
        # Adversarial loss computation
        rnd_pred = tf.random.normal(
            tf.shape(target_pred), stddev=STD_DEV, dtype=target_pred.dtype
        )
        y_pred = discriminator(target_pred + rnd_pred, training=discriminator_training)
        loss_fake = self._loss(
            tf.ones_like(y_pred), y_pred, sample_weight=sample_weight
        )
        loss_fake = tf.cast(loss_fake, dtype=target_pred.dtype)
        return loss_fake
