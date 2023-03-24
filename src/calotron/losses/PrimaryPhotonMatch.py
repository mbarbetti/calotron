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

    def transformer_loss(
        self,
        transformer,
        discriminator,
        source,
        target,
        sample_weight=None,
        training=True,
    ) -> tf.Tensor:
        # Photon-cluster matching weights
        source_coords = tf.tile(source[:, None, :, :2], (1, tf.shape(target)[1], 1, 1))
        target_coords = tf.tile(target[:, :, None, :2], (1, 1, tf.shape(source)[1], 1))
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
        output = transformer((source, target), training=training)
        match_loss = self._mse_loss(target, output, sample_weight=weights)
        match_loss = tf.cast(match_loss, dtype=target.dtype)

        # Adversarial loss
        rnd_pred = tf.random.normal(
            tf.shape(output), stddev=self._noise_stddev, dtype=output.dtype
        )
        y_pred = discriminator(output + rnd_pred, training=False)
        adv_loss = self._bce_loss(
            tf.ones_like(y_pred), y_pred, sample_weight=sample_weight
        )
        adv_loss = tf.cast(adv_loss, dtype=output.dtype)

        # Global event reco loss
        reco_loss = self._mse_loss(
            target[:, :, 2:], output[:, :, 2:], sample_weight=sample_weight
        )
        reco_loss = tf.cast(reco_loss, dtype=target.dtype)
        return match_loss + self._alpha * adv_loss + self._beta * reco_loss

    def discriminator_loss(
        self,
        transformer,
        discriminator,
        source,
        target,
        sample_weight=None,
        training=True,
    ) -> tf.Tensor:
        # Real target loss
        rnd_true = tf.random.normal(
            tf.shape(target), stddev=self._noise_stddev, dtype=target.dtype
        )
        y_true = discriminator(target + rnd_true, training=training)
        real_loss = self._bce_loss(
            tf.ones_like(y_true), y_true, sample_weight=sample_weight
        )
        real_loss = tf.cast(real_loss, dtype=target.dtype)

        # Fake target loss
        output = transformer((source, target), training=False)
        rnd_pred = tf.random.normal(
            tf.shape(output), stddev=self._noise_stddev, dtype=output.dtype
        )
        y_pred = discriminator(output + rnd_pred, training=training)
        fake_loss = self._bce_loss(
            tf.zeros_like(y_pred), y_pred, sample_weight=sample_weight
        )
        fake_loss = tf.cast(fake_loss, dtype=output.dtype)
        return (real_loss + fake_loss) / 2.0

    def aux_classifier_loss(
        self, aux_classifier, source, target, sample_weight=None, training=True
    ) -> tf.Tensor:
        # Photon-cluster matching labels
        source_coords = tf.tile(source[:, :, None, :2], (1, 1, tf.shape(target)[1], 1))
        target_coords = tf.tile(target[:, None, :, :2], (1, tf.shape(source)[1], 1, 1))
        pairwise_distance = tf.norm(
            target_coords - source_coords, ord="euclidean", axis=-1
        )
        pairwise_distance = tf.reduce_min(pairwise_distance, axis=-1)
        labels = tf.cast(
            pairwise_distance < self._max_match_distance, dtype=source.dtype
        )

        # Classification loss
        output = tf.reshape(
            aux_classifier(source, training=training),
            (tf.shape(source)[0], tf.shape(source)[1]),
        )
        clf_loss = self._bce_loss(labels, output, sample_weight=sample_weight)
        clf_loss = tf.cast(clf_loss, dtype=output.dtype)
        return clf_loss

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
