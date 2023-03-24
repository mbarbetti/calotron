import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError as TF_MSE

from calotron.losses.BaseLoss import BaseLoss


class MeanSquaredError(BaseLoss):
    def __init__(self, name="mse_loss") -> None:
        super().__init__(name)
        self._loss = TF_MSE(reduction="auto")

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
        loss = self._loss(y_true, y_pred, sample_weight=sample_weight)
        loss = tf.cast(loss, dtype=target.dtype)
        return loss  # error minimization

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
        loss = self._loss(y_true, y_pred, sample_weight=sample_weight)
        loss = tf.cast(loss, dtype=target.dtype)
        return -loss  # error maximization
