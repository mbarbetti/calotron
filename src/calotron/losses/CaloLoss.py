import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy as TF_BCE
from tensorflow.keras.losses import MeanSquaredError as TF_MSE

from calotron.losses.BaseLoss import BaseLoss


class CaloLoss(BaseLoss):
    def __init__(
        self,
        alpha=0.1,
        from_logits=False,
        label_smoothing=0.0,
        axis=-1,
        reduction="auto",
        name="calo_loss",
    ):
        super().__init__(name)
        self._alpha = float(alpha)
        self._mse_loss = TF_MSE(reduction=reduction)
        self._bce_loss = TF_BCE(
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
        loss_real = self._bce_loss(
            tf.ones_like(y_true), y_true, sample_weight=sample_weight
        )
        loss_fake = self._bce_loss(
            tf.zeros_like(y_pred), y_pred, sample_weight=sample_weight
        )
        return loss_real + loss_fake

    def transformer_loss(
        self, discriminator, target_true, target_pred, sample_weight=None
    ):
        mse_loss = self._mse_loss(target_true, target_pred, sample_weight=sample_weight)
        y_pred = discriminator(target_pred)
        bce_loss = self._bce_loss(
            tf.ones_like(y_pred), y_pred, sample_weight=sample_weight
        )
        return mse_loss + self._alpha * bce_loss
