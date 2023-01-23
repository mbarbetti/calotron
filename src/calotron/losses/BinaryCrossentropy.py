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
    ):
        super().__init__(name)
        self._loss = TF_BCE(
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            axis=axis,
            reduction=reduction,
        )

    def discriminator_loss(
        self, discriminator, target_true, target_pred, sample_weight=None
    ):
        y_true = discriminator(target_true)
        y_pred = discriminator(target_pred)
        loss_real = self._loss(
            tf.ones_like(y_true), y_true, sample_weight=sample_weight
        )
        loss_fake = self._loss(
            tf.zeros_like(y_pred), y_pred, sample_weight=sample_weight
        )
        return loss_real + loss_fake

    def transformer_loss(
        self, discriminator, target_true, target_pred, sample_weight=None
    ):
        y_pred = discriminator(target_pred)
        loss_fake = self._loss(
            tf.ones_like(y_pred), y_pred, sample_weight=sample_weight
        )
        return loss_fake
