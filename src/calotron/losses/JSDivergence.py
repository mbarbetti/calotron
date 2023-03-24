import tensorflow as tf
from tensorflow.keras.losses import KLDivergence as TF_KLDivergence

from calotron.losses.BaseLoss import BaseLoss


class JSDivergence(BaseLoss):
    def __init__(self, name="js_loss") -> None:
        super().__init__(name)
        self._kl_div = TF_KLDivergence(reduction="auto")

    def transformer_loss(
        self,
        transformer,
        discriminator,
        source,
        target,
        sample_weight=None,
        training=True,
    ) -> tf.Tensor:
        output = transformer((source, target), training=training)
        y_true = discriminator(target, training=False)
        y_pred = discriminator(output, training=False)
        loss = self._js_div(y_true, y_pred, sample_weight=sample_weight)
        loss = tf.cast(loss, dtype=target.dtype)
        return loss  # divergence minimization

    def discriminator_loss(
        self,
        transformer,
        discriminator,
        source,
        target,
        sample_weight=None,
        training=True,
    ) -> tf.Tensor:
        output = transformer((source, target), training=False)
        y_true = discriminator(target, training=training)
        y_pred = discriminator(output, training=training)
        loss = self._js_div(y_true, y_pred, sample_weight=sample_weight)
        loss = tf.cast(loss, dtype=target.dtype)
        return -loss  # divergence maximization

    def _js_div(self, y_true, y_pred, sample_weight=None) -> tf.Tensor:
        loss = 0.5 * self._kl_div(
            y_true, 0.5 * (y_true + y_pred), sample_weight=sample_weight
        ) + 0.5 * self._kl_div(
            y_pred, 0.5 * (y_true + y_pred), sample_weight=sample_weight
        )
        return loss
