import tensorflow as tf

from calotron.losses.BaseLoss import BaseLoss
from calotron.losses.BinaryCrossentropy import BinaryCrossentropy
from calotron.losses.WassersteinDistance import WassersteinDistance

ADV_METRICS = ["binary-crossentropy", "wasserstein-distance"]


class GlobalEventReco(BaseLoss):
    def __init__(
        self,
        lambda_adv=0.1,
        adversarial_metric="binary-crossentropy",
        bce_options={
            "injected_noise_stddev": 0.0,
            "from_logits": False,
            "label_smoothing": 0.0,
        },
        wass_options={"lipschitz_penalty": 100.0, "virtual_direction_upds": 1},
        name="global_evt_reco_loss",
    ) -> None:
        super().__init__(name)

        # Adversarial strength
        assert isinstance(lambda_adv, (int, float))
        assert lambda_adv >= 0.0
        self._lambda_adv = float(lambda_adv)

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
        if self._adversarial_metric == "binary-crossentropy":
            self._adv_loss = BinaryCrossentropy(**self._bce_options)
        elif self._adversarial_metric == "wasserstein-distance":
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
        # Global event reco loss
        output = transformer((source, target), training=training)
        errors = tf.reduce_sum((target - output) ** 2, axis=-1)
        if sample_weight is not None:
            if len(tf.shape(sample_weight)) != len(tf.shape(errors)):
                sample_weight = tf.reduce_mean(sample_weight, axis=-1)
            errors = tf.reduce_sum(sample_weight * errors, axis=-1) / tf.reduce_sum(
                sample_weight, axis=-1
            )
        reco_loss = tf.reduce_mean(errors)
        reco_loss = tf.cast(reco_loss, dtype=target.dtype)

        # Adversarial loss
        adv_loss = self._adv_loss.transformer_loss(
            transformer=transformer,
            discriminator=discriminator,
            source=source,
            target=target,
            sample_weight=sample_weight,
            training=training,
        )

        tot_loss = (
            reco_loss 
            + self._lambda_adv * adv_loss
        )
        return tot_loss

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

    @property
    def lambda_adv(self) -> float:
        return self._lambda_adv

    @property
    def adversarial_metric(self) -> str:
        return self._adversarial_metric

    @property
    def bce_options(self) -> dict:
        return self._bce_options

    @property
    def wass_options(self) -> dict:
        return self._wass_options
