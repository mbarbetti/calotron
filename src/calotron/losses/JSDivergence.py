import tensorflow as tf
from tensorflow.keras.losses import KLDivergence as TF_KLDivergence

from calotron.losses.BaseLoss import BaseLoss


class JSDivergence(BaseLoss):
    def __init__(self, reduction="auto", name="js_loss"):
        super().__init__(name)
        self._kl_div = TF_KLDivergence(reduction=reduction)

    def discriminator_loss(
        self, discriminator, target_true, target_pred, sample_weight=None, training=True
    ):
        y_true = discriminator(target_true, training=training)
        y_pred = discriminator(target_pred, training=training)
        return -self._js_div(
            y_true, y_pred, sample_weight=sample_weight
        )  # divergence maximization

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
        return self._js_div(
            y_true, y_pred, sample_weight=sample_weight
        )  # divergence minimization

    def _js_div(self, y_true, y_pred, sample_weight=None):
        dtype = self._kl_div(y_true, y_pred).dtype
        y_true = tf.cast(y_true, dtype=dtype)
        y_pred = tf.cast(y_pred, dtype=dtype)
        loss = 0.5 * self._kl_div(
            y_true, 0.5 * (y_true + y_pred), sample_weight=sample_weight
        ) + 0.5 * self._kl_div(
            y_pred, 0.5 * (y_true + y_pred), sample_weight=sample_weight
        )
        return loss
