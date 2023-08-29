import tensorflow as tf

from calotron.losses.BaseLoss import BaseLoss
from calotron.losses.BinaryCrossentropy import BinaryCrossentropy
from calotron.losses.WassersteinDistance import WassersteinDistance

ADV_METRICS = ["binary-crossentropy", "wasserstein-distance"]


class AdvLoss(BaseLoss):
    def __init__(
        self,
        alpha=0.5,
        adversarial_metric="binary-crossentropy",
        bce_options={
            "injected_noise_stddev": 0.0,
            "from_logits": False,
            "label_smoothing": 0.0,
        },
        wass_options={
            "lipschitz_regularizer": "alp",
            "lipschitz_penalty": 100.0,
            "lipschitz_penalty_strategy": "one-sided",
        },
        warmup_energy=0.0,
        name="mse_loss",
    ) -> None:
        super().__init__(name)

        # Adversarial scale
        assert isinstance(alpha, (int, float))
        assert alpha >= 0.0 and alpha < 1.0
        self._alpha = float(alpha)
        self._gamma = tf.Variable(self._alpha / (1 - self._alpha), name="gamma")

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

        # Warmup energy
        assert isinstance(warmup_energy, (int, float))
        assert warmup_energy >= 0.0
        self._warmup_energy = float(warmup_energy)

        for options in [self._bce_options, self._wass_options]:
            options.update(dict(warmup_energy=warmup_energy))

        # Adversarial loss definition
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
        raise NotImplementedError(
            "Only `AdvLoss` subclasses have the "
            "`transformer_loss()` method implemented."
        )

    @staticmethod
    def _compute_adv_loss(main_loss, adv_loss, gamma=0.5) -> tf.Tensor:
        main_scale = tf.math.round(tf.math.log(tf.abs(main_loss)) / tf.math.log(10.0))
        adv_scale = tf.math.round(tf.math.log(tf.abs(adv_loss)) / tf.math.log(10.0))
        scale = tf.stop_gradient(10 ** (main_scale - adv_scale))
        tot_loss = main_loss + gamma * scale * adv_loss
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
    def alpha(self) -> float:
        return self._alpha

    @property
    def gamma(self) -> tf.Variable:
        return self._gamma

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
    def warmup_energy(self) -> float:
        return self._warmup_energy
