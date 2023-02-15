import tensorflow as tf
from tensorflow.keras.losses import KLDivergence as TF_KLDivergence

from calotron.losses.BaseLoss import BaseLoss


class JSDivergence(BaseLoss):
    def __init__(self, reduction="auto", name="js_loss"):
        super().__init__(name)
        self._kl_div = TF_KLDivergence(reduction=reduction)

    def discriminator_loss(
        self, discriminator, target_true, target_pred, sample_weight=None
    ):
        y_true = discriminator(target_true, training=True)
        y_pred = discriminator(target_pred, training=True)
        loss = self._js_div(y_true, y_pred, sample_weight=sample_weight)
        loss = tf.cast(loss, dtype=target_true.dtype)
        return -loss  # divergence maximization

    def transformer_loss(
        self, discriminator, target_true, target_pred, sample_weight=None
    ):
        y_true = discriminator(target_true, training=False)
        y_pred = discriminator(target_pred, training=False)
        loss = self._js_div(y_true, y_pred, sample_weight=sample_weight)
        loss = tf.cast(loss, dtype=target_true.dtype)
        return loss  # divergence minimization

    def _js_div(self, y_true, y_pred, sample_weight=None):
        loss = 0.5 * self._kl_div(
            y_true, 0.5 * (y_true + y_pred), sample_weight=sample_weight
        ) + 0.5 * self._kl_div(
            y_pred, 0.5 * (y_true + y_pred), sample_weight=sample_weight
        )
        return loss
