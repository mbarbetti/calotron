import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy as TF_BCE
from tensorflow.keras.losses import MeanSquaredError as TF_MSE

from calotron.losses.BaseLoss import BaseLoss

STD_DEV = 0.02


class EuclideanWeightedError(BaseLoss):
    def __init__(
        self,
        alpha=0.1,
        beta=1.0,
        max_match_distance=0.01,
        discriminator_from_logits=False,
        discriminator_label_smoothing=0.0,
        name="ewe_loss",
    ) -> None:
        super().__init__(name)

        # Balance btw EWE and BCE
        assert isinstance(alpha, (int, float))
        assert alpha >= 0.0
        self._alpha = float(alpha)

        # Balance btw EWE and MSE
        assert isinstance(beta, (int, float))
        assert beta >= 0.0
        self._beta = float(beta)

        # Max distance for matching
        assert isinstance(max_match_distance, (int, float))
        assert max_match_distance > 0.0
        self._max_match_distance = max_match_distance

        # TensorFlow BinaryCrossentropy
        self._bce_loss = TF_BCE(
            from_logits=discriminator_from_logits,
            label_smoothing=discriminator_label_smoothing,
            axis=-1,
            reduction="auto",
        )

        # TensorFlow MeanSquaredError
        self._mse_loss = TF_MSE(reduction="auto")

    def discriminator_loss(
        self,
        discriminator,
        source_true,
        target_true,
        target_pred,
        sample_weight=None,
        discriminator_training=True,
    ) -> tf.Tensor:
        # Real loss computation
        rnd_true = tf.random.normal(
            tf.shape(target_true), stddev=STD_DEV, dtype=target_true.dtype
        )
        y_true = discriminator(target_true + rnd_true, training=discriminator_training)
        loss_real = self._bce_loss(
            tf.ones_like(y_true), y_true, sample_weight=sample_weight
        )
        loss_real = tf.cast(loss_real, dtype=target_true.dtype)

        # Fake loss computation
        rnd_pred = tf.random.normal(
            tf.shape(target_pred), stddev=STD_DEV, dtype=target_pred.dtype
        )
        y_pred = discriminator(target_pred + rnd_pred, training=discriminator_training)
        loss_fake = self._bce_loss(
            tf.zeros_like(y_pred), y_pred, sample_weight=sample_weight
        )
        loss_fake = tf.cast(loss_fake, dtype=target_pred.dtype)
        return (loss_real + loss_fake) / 2.0

    def transformer_loss(
        self,
        discriminator,
        source_true,
        target_true,
        target_pred,
        sample_weight=None,
        discriminator_training=False,
    ) -> tf.Tensor:
        # Euclidean weights computation
        source_coords = tf.tile(
            source_true[:, None, :, :2], (1, tf.shape(target_true)[1], 1, 1)
        )
        target_coords = tf.tile(
            target_true[:, :, None, :2], (1, 1, tf.shape(source_true)[1], 1)
        )
        match_distance = tf.norm(
            target_coords - source_coords, ord="euclidean", axis=-1
        )
        match_distance = tf.reduce_min(match_distance, axis=-1)
        weights = self._max_match_distance / tf.math.maximum(
            match_distance, self._max_match_distance
        )
        if sample_weight is not None:
            weights = tf.math.multiply(weights, sample_weight)

        # Regression loss computation
        target_true_ = tf.tile(
            target_true[:, :, None, :], (1, 1, tf.shape(target_pred)[1], 1)
        )
        target_pred_ = tf.tile(
            target_pred[:, None, :, :], (1, tf.shape(target_true)[1], 1, 1)
        )
        errors = tf.norm(target_pred_ - target_true_, ord="euclidean", axis=-1)
        errors = tf.reduce_min(errors, axis=-1)
        reg_loss = tf.reduce_sum(tf.math.multiply(weights, errors)) / tf.reduce_sum(
            weights
        )
        reg_loss = tf.cast(reg_loss, dtype=target_true.dtype)

        # Adversarial loss computation
        rnd_pred = tf.random.normal(
            tf.shape(target_pred), stddev=STD_DEV, dtype=target_pred.dtype
        )
        y_pred = discriminator(target_pred + rnd_pred, training=discriminator_training)
        adv_loss = self._bce_loss(
            tf.ones_like(y_pred), y_pred, sample_weight=sample_weight
        )
        adv_loss = tf.cast(adv_loss, dtype=target_pred.dtype)

        # MSE loss computation
        mse_loss = self._mse_loss(target_true, target_pred, sample_weight=sample_weight)
        mse_loss = tf.cast(mse_loss, dtype=target_true.dtype)
        return reg_loss + self._alpha * adv_loss + self._beta * mse_loss
