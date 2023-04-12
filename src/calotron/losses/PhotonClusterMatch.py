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
        lambda_adv=0.1,
        lambda_geom=1.0,
        lambda_global=0.5,
        max_match_distance=1e-4,
        adversarial_metric="binary-crossentropy",
        bce_options={
            "injected_noise_stddev": 0.0,
            "from_logits": False,
            "label_smoothing": 0.0,
        },
        wass_options={"lipschitz_penalty": 100.0, "virtual_direction_upds": 1},
        aux_bce_options={"from_logits": False, "label_smoothing": 0.0},
        name="photon_cluster_match_loss",
    ) -> None:
        super().__init__(name)

        # Adversarial strength
        assert isinstance(lambda_adv, (int, float))
        assert lambda_adv >= 0.0
        self._lambda_adv = float(lambda_adv)

        # Geometrical width strength
        assert isinstance(lambda_geom, (int, float))
        assert lambda_geom >= 0.0
        self._lambda_geom = float(lambda_geom)

        # Global event reco strength
        assert isinstance(lambda_global, (int, float))
        assert lambda_global >= 0.0
        self._lambda_global = float(lambda_global)

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
        assert isinstance(aux_bce_options, dict)
        self._aux_bce_options = aux_bce_options

        # Losses definition
        if adversarial_metric == "binary-crossentropy":
            self._adv_loss = BinaryCrossentropy(**self._bce_options)
        elif adversarial_metric == "wasserstein-distance":
            self._adv_loss = WassersteinDistance(**self._wass_options)
        self._aux_loss = TF_BCE(**self._aux_bce_options)

    def transformer_loss(
        self,
        transformer,
        discriminator,
        source,
        target,
        sample_weight=None,
        training=True,
    ) -> tf.Tensor:
        # Photon-cluster matching loss
        output = transformer((source, target), training=training)
        match_weights = self._matching_weights(source, target, sample_weight)
        match_err = tf.reduce_sum((target - output) ** 2, axis=-1)
        match_err = tf.reduce_sum(match_weights * match_err, axis=-1) / tf.reduce_sum(
            match_weights, axis=-1
        )
        match_loss = tf.reduce_mean(match_err, axis=-1)
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

        # Geometrical width loss
        target_width = tf.math.reduce_std(target[:, :, :2], axis=1)  # per event
        output_width = tf.math.reduce_std(output[:, :, :2], axis=1)  # per event
        geom_err = tf.reduce_sum((target_width - output_width) ** 2, axis=-1)
        if sample_weight is not None:
            if len(tf.shape(sample_weight)) != len(tf.shape(geom_err)):
                weights = tf.reduce_mean(sample_weight, axis=[1, 2])
            geom_loss = tf.reduce_sum(weights * geom_err, axis=-1) / tf.reduce_sum(
                weights, axis=-1
            )
        else:
            geom_loss = tf.reduce_mean(geom_err)
        geom_loss = tf.cast(geom_loss, dtype=target.dtype)

        # Global event reco loss
        global_err = tf.reduce_sum((target[:, :, 2:] - output[:, :, 2:]) ** 2, axis=-1)
        if sample_weight is not None:
            if len(tf.shape(sample_weight)) != len(tf.shape(global_err)):
                weights = tf.reduce_mean(sample_weight, axis=-1)
            global_err = tf.reduce_sum(weights * global_err, axis=-1) / tf.reduce_sum(
                weights, axis=-1
            )
        global_loss = tf.reduce_mean(global_err)
        global_loss = tf.cast(global_loss, dtype=target.dtype)

        tot_loss = (
            match_loss
            + self._lambda_adv * adv_loss
            + self._lambda_geom * geom_loss
            + self._lambda_global * global_loss
        )
        return tot_loss

    def _matching_weights(self, source, target, sample_weight=None) -> tf.Tensor:
        source_xy = tf.tile(source[:, None, :, :2], (1, tf.shape(target)[1], 1, 1))
        target_xy = tf.tile(target[:, :, None, :2], (1, 1, tf.shape(source)[1], 1))
        pairwise_distance = tf.norm(target_xy - source_xy, axis=-1)
        pairwise_distance = tf.reduce_min(pairwise_distance, axis=-1)
        match_weights = self._max_match_distance / tf.math.maximum(
            pairwise_distance, self._max_match_distance
        )
        if sample_weight is not None:
            if len(tf.shape(sample_weight)) != len(tf.shape(match_weights)):
                sample_weight = tf.reduce_mean(sample_weight, axis=-1)
            match_weights *= sample_weight
        return match_weights

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
        pairwise_distance = tf.norm(target_coords - source_coords, axis=-1)
        pairwise_distance = tf.reduce_min(pairwise_distance, axis=-1)
        labels = tf.cast(
            pairwise_distance < self._max_match_distance, dtype=source.dtype
        )

        # Classification loss
        output = tf.reshape(
            aux_classifier(source, training=training),
            (tf.shape(source)[0], tf.shape(source)[1]),
        )
        clf_loss = self._aux_loss(labels, output, sample_weight=evt_weights)
        clf_loss = tf.cast(clf_loss, dtype=output.dtype)
        return clf_loss

    @property
    def lambda_adv(self) -> float:
        return self._lambda_adv

    @property
    def lambda_geom(self) -> float:
        return self._lambda_geom

    @property
    def lambda_global(self) -> float:
        return self._lambda_global

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

    @property
    def aux_bce_options(self) -> dict:
        return self._aux_bce_options
