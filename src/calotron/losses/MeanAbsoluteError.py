import tensorflow as tf
from tensorflow.keras.losses import MeanAbsoluteError as TF_MAE

from calotron.losses.BaseLoss import BaseLoss


class MeanAbsoluteError(BaseLoss):
    def __init__(self, reduction="auto", name="mae_loss"):
        super().__init__(name)
        self._loss = TF_MAE(reduction=reduction)

    def discriminator_loss(
        self, discriminator, target_true, target_pred, sample_weight=None, training=True
    ):
        y_true = discriminator(target_true, training=training)
        y_pred = discriminator(target_pred, training=training)
        return -self._loss(
            y_true, y_pred, sample_weight=sample_weight
        )  # error maximization

    def transformer_loss(
        self,
        discriminator,
        target_true,
        target_pred,
        sample_weight=None,
        training=False,
    ):
        y_true = discriminator(target_true, training=training)
        y_pred = discriminator(target_pred, training=training)
        return self._loss(
            y_true, y_pred, sample_weight=sample_weight
        )  # error minimization
