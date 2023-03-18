import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy as TF_BCE
from tensorflow.keras.losses import MeanSquaredError as TF_MSE

from calotron.losses.BaseLoss import BaseLoss


class PrimaryPhotonMatch(BaseLoss):
    def __init__(
        self,
        alpha=0.1,
        beta=0.0,
        max_match_distance=0.01,
        noise_stddev=0.05,
        from_logits=False,
        label_smoothing=0.0,
        name="photon_match_loss",
    ) -> None:
        super().__init__(name)

        # Adversarial strength
        assert isinstance(alpha, (int, float))
        assert alpha >= 0.0
        self._alpha = float(alpha)

        # Global event reco strength
        assert isinstance(beta, (int, float))
        assert beta >= 0.0
        self._beta = float(beta)

        # Max distance for photon-cluster matching
        assert isinstance(max_match_distance, (int, float))
        assert max_match_distance > 0.0
        self._max_match_distance = max_match_distance

        # Noise standard deviation
        assert isinstance(noise_stddev, (int, float))
        assert noise_stddev > 0.0
        self._noise_stddev = noise_stddev

        # BCE `from_logits` flag
        assert isinstance(from_logits, bool)
        self._from_logits = from_logits

        # BCE `label_smoothing`
        assert isinstance(label_smoothing, (int, float))
        assert label_smoothing >= 0.0 and label_smoothing <= 1.0
        self._label_smoothing = label_smoothing

        # TensorFlow MeanSquaredError
        self._mse_loss = TF_MSE(reduction="auto")

        # TensorFlow BinaryCrossentropy
        self._bce_loss = TF_BCE(
            from_logits=self._from_logits,
            label_smoothing=self._label_smoothing,
            axis=-1,
            reduction="auto",
        )

    def discriminator_loss(
        self,
        discriminator,
        source_true,
        target_true,
        target_pred,
        sample_weight=None,
        discriminator_training=True,
    ) -> tf.Tensor:
        # Real target loss
        rnd_true = tf.random.normal(
            tf.shape(target_true), stddev=self._noise_stddev, dtype=target_true.dtype
        )
        y_true = discriminator(target_true + rnd_true, training=discriminator_training)
        real_loss = self._bce_loss(
            tf.ones_like(y_true), y_true, sample_weight=sample_weight
        )
        real_loss = tf.cast(real_loss, dtype=target_true.dtype)

        # Fake target loss
        rnd_pred = tf.random.normal(
            tf.shape(target_pred), stddev=self._noise_stddev, dtype=target_pred.dtype
        )
        y_pred = discriminator(target_pred + rnd_pred, training=discriminator_training)
        fake_loss = self._bce_loss(
            tf.zeros_like(y_pred), y_pred, sample_weight=sample_weight
        )
        fake_loss = tf.cast(fake_loss, dtype=target_pred.dtype)
        return (real_loss + fake_loss) / 2.0

    def transformer_loss(
        self,
        discriminator,
        source_true,
        target_true,
        target_pred,
        sample_weight=None,
        discriminator_training=False,
    ) -> tf.Tensor:
        # Photon-cluster matching weights
        source_coords = tf.tile(
            source_true[:, None, :, :2], (1, tf.shape(target_true)[1], 1, 1)
        )
        target_coords = tf.tile(
            target_true[:, :, None, :2], (1, 1, tf.shape(source_true)[1], 1)
        )
        pairwise_distance = tf.norm(
            target_coords - source_coords, ord="euclidean", axis=-1
        )
        pairwise_distance = tf.reduce_min(pairwise_distance, axis=-1)
        weights = self._max_match_distance / tf.math.maximum(
            pairwise_distance, self._max_match_distance
        )
        if sample_weight is not None:
            weights *= sample_weight

        # Photon-cluster matching loss
        match_loss = self._mse_loss(target_true, target_pred, sample_weight=weights)
        match_loss = tf.cast(match_loss, dtype=target_true.dtype)

        # Adversarial loss
        rnd_pred = tf.random.normal(
            tf.shape(target_pred), stddev=self._noise_stddev, dtype=target_pred.dtype
        )
        y_pred = discriminator(target_pred + rnd_pred, training=discriminator_training)
        adv_loss = self._bce_loss(
            tf.ones_like(y_pred), y_pred, sample_weight=sample_weight
        )
        adv_loss = tf.cast(adv_loss, dtype=target_pred.dtype)

        # Global event reco loss
        reco_loss = self._mse_loss(
            target_true[:, :, 2:], target_pred[:, :, 2:], sample_weight=sample_weight
        )
        reco_loss = tf.cast(reco_loss, dtype=target_true.dtype)
        return match_loss + self._alpha * adv_loss + self._beta * reco_loss

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def max_match_distance(self) -> float:
        return self._max_match_distance

    @property
    def noise_stddev(self) -> float:
        return self._noise_stddev

    @property
    def from_logits(self) -> bool:
        return self._from_logits

    @property
    def label_smoothing(self) -> float:
        return self._label_smoothing
