import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError as TF_MSE

from calotron.losses.BaseLoss import BaseLoss


class MeanSquaredError(BaseLoss):
    def __init__(self, ignore_padding=False, name="mse_loss") -> None:
        super().__init__(name)

        # Ignore padding
        assert isinstance(ignore_padding, bool)
        self._ignore_padding = ignore_padding

        # MSE
        self._loss = TF_MSE()

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

        if self._ignore_padding:
            mask = tf.cast(target[:, :, 2] > 0.0, dtype=target.dtype)  # not padded values
        else:
            mask = None
        y_true = discriminator((source, target), mask=mask, training=False)
        y_pred = discriminator((source, output), mask=mask, training=False)

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

        if self._ignore_padding:
            mask = tf.cast(target[:, :, 2] > 0.0, dtype=target.dtype)  # not padded values
        else:
            mask = None
        y_true = discriminator((source, target), mask=mask, training=training)
        y_pred = discriminator((source, output), mask=mask, training=training)

        if sample_weight is not None:
            evt_weights = tf.reduce_mean(sample_weight, axis=1)
        else:
            evt_weights = None
        loss = self._loss(y_true, y_pred, sample_weight=evt_weights)
        loss = tf.cast(loss, dtype=target.dtype)
        return -loss  # error maximization
    
    @property
    def ignore_padding(self) -> bool:
        return self._ignore_padding
