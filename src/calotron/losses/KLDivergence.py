import tensorflow as tf
from tensorflow.keras.losses import KLDivergence as TF_KLDivergence

from calotron.losses.BaseLoss import BaseLoss


class KLDivergence(BaseLoss):
    def __init__(self, reduction="auto", name="kl_loss") -> None:
        super().__init__(name)
        self._loss = TF_KLDivergence(reduction=reduction)

    def discriminator_loss(
        self,
        discriminator,
        target_true,
        target_pred,
        sample_weight=None,
        discriminator_training=True,
    ) -> tf.Tensor:
        y_true = discriminator(target_true, training=discriminator_training)
        y_pred = discriminator(target_pred, training=discriminator_training)
        loss = self._loss(y_true, y_pred, sample_weight=sample_weight)
        loss = tf.cast(loss, dtype=target_true.dtype)
        return -loss  # divergence maximization

    def transformer_loss(
        self,
        discriminator,
        target_true,
        target_pred,
        sample_weight=None,
        discriminator_training=False,
    ) -> tf.Tensor:
        y_true = discriminator(target_true, training=discriminator_training)
        y_pred = discriminator(target_pred, training=discriminator_training)
        loss = self._loss(y_true, y_pred, sample_weight=sample_weight)
        loss = tf.cast(loss, dtype=target_true.dtype)
        return loss  # divergence minimization
