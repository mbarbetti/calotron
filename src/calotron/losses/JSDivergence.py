import tensorflow as tf
from tensorflow.keras.losses import KLDivergence as TF_KLDivergence

from calotron.losses.BaseLoss import BaseLoss


class JSDivergence(BaseLoss):
    def __init__(self, ignore_padding=False, name="js_loss") -> None:
        super().__init__(name)

        # Ignore padding
        assert isinstance(ignore_padding, bool)
        self._ignore_padding = ignore_padding

        # K-L divergence
        self._kl_div = TF_KLDivergence()

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
            source_mask = tf.cast(
                (source[:, :, 0] != 0.0) & (source[:, :, 1] != 0.0), dtype=target.dtype
            )
            target_mask = tf.cast(
                (target[:, :, 0] != 0.0) & (target[:, :, 1] != 0.0), dtype=target.dtype
            )
            mask = (source_mask, target_mask)
        else:
            mask = None
        y_true = discriminator((source, target), mask=mask, training=False)
        y_pred = discriminator((source, output), mask=mask, training=False)

        if sample_weight is not None:
            evt_weights = tf.reduce_mean(sample_weight, axis=1)
        else:
            evt_weights = None
        loss = self._js_div(y_true, y_pred, sample_weight=evt_weights)
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

        if self._ignore_padding:
            source_mask = tf.cast(
                (source[:, :, 0] != 0.0) & (source[:, :, 1] != 0.0), dtype=target.dtype
            )
            target_mask = tf.cast(
                (target[:, :, 0] != 0.0) & (target[:, :, 1] != 0.0), dtype=target.dtype
            )
            mask = (source_mask, target_mask)
        else:
            mask = None
        y_true = discriminator((source, target), mask=mask, training=training)
        y_pred = discriminator((source, output), mask=mask, training=training)

        if sample_weight is not None:
            evt_weights = tf.reduce_mean(sample_weight, axis=1)
        else:
            evt_weights = None
        loss = self._js_div(y_true, y_pred, sample_weight=evt_weights)
        loss = tf.cast(loss, dtype=target.dtype)
        return -loss  # divergence maximization

    def _js_div(self, y_true, y_pred, sample_weight=None) -> tf.Tensor:
        loss = 0.5 * self._kl_div(
            y_true, 0.5 * (y_true + y_pred), sample_weight=sample_weight
        ) + 0.5 * self._kl_div(
            y_pred, 0.5 * (y_true + y_pred), sample_weight=sample_weight
        )
        return loss

    @property
    def ignore_padding(self) -> bool:
        return self._ignore_padding
