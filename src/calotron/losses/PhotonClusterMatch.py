import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy as TF_BCE
from tensorflow.keras.losses import MeanSquaredError as TF_MSE

from calotron.losses.BaseLoss import BaseLoss
from calotron.losses.BinaryCrossentropy import BinaryCrossentropy
from calotron.losses.WassersteinDistance import WassersteinDistance

ADV_METRICS = ["binary-crossentropy", "wasserstein-distance"]
DEFAULT_DISTANCE = 99.0


class PhotonClusterMatch(BaseLoss):
    def __init__(
        self,
        alpha=0.1,
        beta=0.0,
        max_match_distance=5e-3,
        adversarial_metric="binary-crossentropy",
        bce_options={
            "injected_noise_stddev": 0.0,
            "from_logits": False,
            "label_smoothing": 0.0,
        },
        wass_options={"lipschitz_penalty": 100.0, "virtual_direction_upds": 1},
        name="photon_cluster_match_loss",
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
        self._max_match_distance = float(max_match_distance)

        # Adversarial metric
        assert isinstance(adversarial_metric, str)
        if adversarial_metric not in ADV_METRICS:
            raise ValueError(
                "`adversarial_metric` should be selected "
                f"in {ADV_METRICS}, instead "
                f"'{adversarial_metric}' passed"
            )
        self._adversarial_metric = adversarial_metric

        # Options
        assert isinstance(bce_options, dict)
        self._bce_options = bce_options
        assert isinstance(wass_options, dict)
        self._wass_options = wass_options

        # Losses definition
        self._mse_loss = TF_MSE()
        tf_bce_options = self._bce_options.copy()
        if "injected_noise_stddev" in tf_bce_options.keys():
            tf_bce_options.pop("injected_noise_stddev")
        self._bce_loss = TF_BCE(**tf_bce_options)
        if adversarial_metric == "binary-crossentropy":
            self._adv_loss = BinaryCrossentropy(**self._bce_options)
        elif adversarial_metric == "wasserstein-distance":
            self._adv_loss = WassersteinDistance(**self._wass_options)

    def transformer_loss(
        self,
        transformer,
        discriminator,
        source,
        target,
        sample_weight=None,
        training=True,
    ) -> tf.Tensor:
        if sample_weight is not None:
            evt_weights = tf.reduce_mean(sample_weight, axis=1)
        else:
            evt_weights = None

        # Photon-cluster matching loss
        output = transformer((source, target), training=training)
        match_weights = self._matching_weights(source, target, evt_weights)
        errors = tf.reduce_sum((target - output) ** 2, axis=-1)
        errors = tf.reduce_sum(match_weights * errors, axis=-1) / tf.reduce_sum(
            match_weights, axis=-1
        )
        match_loss = tf.reduce_mean(errors, axis=-1)
        match_loss = tf.cast(match_loss, dtype=target.dtype)

        # Adversarial loss
        adv_loss = self._adv_loss.transformer_loss(
            transformer=transformer,
            discriminator=discriminator,
            source=source,
            target=target,
            sample_weight=sample_weight,
            training=training,
        )

        # Global event reco loss
        reco_loss = self._mse_loss(
            target[:, :, 2:], output[:, :, 2:], sample_weight=evt_weights
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
        adv_loss = self._adv_loss.discriminator_loss(
            transformer=transformer,
            discriminator=discriminator,
            source=source,
            target=target,
            sample_weight=sample_weight,
            training=training,
        )
        return adv_loss

    def aux_classifier_loss(
        self, aux_classifier, source, target, sample_weight=None, training=True
    ) -> tf.Tensor:
        if sample_weight is not None:
            evt_weights = tf.reduce_mean(sample_weight, axis=1)
        else:
            evt_weights = None

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
        clf_loss = self._bce_loss(labels, output, sample_weight=evt_weights)
        clf_loss = tf.cast(clf_loss, dtype=output.dtype)
        return clf_loss

    def _matching_weights(self, source, target, weights=None) -> tf.Tensor:
        source_xy = tf.tile(source[:, None, :, :2], (1, tf.shape(target)[1], 1, 1))
        target_xy = tf.tile(target[:, :, None, :2], (1, 1, tf.shape(source)[1], 1))
        pairwise_distance = tf.norm(target_xy - source_xy, ord="euclidean", axis=-1)
        pairwise_distance = tf.reduce_min(
            tf.where(pairwise_distance > 0.0, pairwise_distance, DEFAULT_DISTANCE),
            axis=-1,
        )  # ignore padded values
        evt_weights = self._max_match_distance / tf.math.maximum(
            pairwise_distance, self._max_match_distance
        )
        if weights is not None:
            evt_weights *= tf.reduce_mean(weights, axis=1)[:, None]
        return evt_weights

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
    def adversarial_metric(self) -> str:
        return self._adversarial_metric

    @property
    def bce_options(self) -> dict:
        return self._bce_options

    @property
    def wass_options(self) -> dict:
        return self._wass_options
