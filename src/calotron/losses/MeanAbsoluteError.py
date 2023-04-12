import tensorflow as tf
from tensorflow.keras.losses import MeanAbsoluteError as TF_MAE

from calotron.losses.BaseLoss import BaseLoss


class MeanAbsoluteError(BaseLoss):
    def __init__(self, name="mae_loss") -> None:
        super().__init__(name)
        self._loss = TF_MAE()

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
        y_true = discriminator((source, target), training=False)
        y_pred = discriminator((source, output), training=False)

        if sample_weight is not None:
            evt_weights = tf.reduce_mean(sample_weight, axis=1)
        else:
            evt_weights = None
        loss = self._loss(y_true, y_pred, sample_weight=evt_weights)
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
        y_true = discriminator((source, target), training=training)
        y_pred = discriminator((source, output), training=training)

        if sample_weight is not None:
            evt_weights = tf.reduce_mean(sample_weight, axis=1)
        else:
            evt_weights = None
        loss = self._loss(y_true, y_pred, sample_weight=evt_weights)
        loss = tf.cast(loss, dtype=target.dtype)
        return -loss  # error maximization
