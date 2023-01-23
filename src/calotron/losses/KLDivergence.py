import tensorflow as tf
from tensorflow.keras.losses import KLDivergence as TF_KLDivergence

from calotron.losses.BaseLoss import BaseLoss


class KLDivergence(BaseLoss):
    def __init__(self, reduction="auto", name="kl_loss"):
        super().__init__(name)
        self._loss = TF_KLDivergence(reduction=reduction)

    def discriminator_loss(
        self, discriminator, target_true, target_pred, sample_weight=None
    ):
        y_true = discriminator(target_true)
        y_pred = discriminator(target_pred)
        return -self._loss(
            y_true, y_pred, sample_weight=sample_weight
        )  # divergence maximization

    def transformer_loss(
        self, discriminator, target_true, target_pred, sample_weight=None
    ):
        y_true = discriminator(target_true)
        y_pred = discriminator(target_pred)
        return self._loss(
            y_true, y_pred, sample_weight=sample_weight
        )  # divergence minimization
