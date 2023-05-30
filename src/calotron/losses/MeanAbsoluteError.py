import tensorflow as tf
from tensorflow.keras.losses import MeanAbsoluteError as TF_MAE

from calotron.losses.BaseLoss import BaseLoss
from calotron.losses.BinaryCrossentropy import BinaryCrossentropy
from calotron.losses.WassersteinDistance import WassersteinDistance

ADV_METRICS = ["binary-crossentropy", "wasserstein-distance"]


class MeanAbsoluteError(BaseLoss):
    def __init__(
        self,
        warmup_energy=0.0,
        alpha=0.1,
        adversarial_metric="binary-crossentropy",
        bce_options={
            "injected_noise_stddev": 0.0,
            "from_logits": False,
            "label_smoothing": 0.0,
        },
        wass_options={"lipschitz_penalty": 100.0, "virtual_direction_upds": 1},
        name="mae_loss",
    ) -> None:
        super().__init__(name)

        # Warmup energy
        assert isinstance(warmup_energy, (int, float))
        assert warmup_energy >= 0.0
        self._warmup_energy = float(warmup_energy)

        # Adversarial scale
        assert isinstance(alpha, (int, float))
        assert alpha >= 0.0
        self._init_alpha = float(alpha)
        self._alpha = tf.Variable(float(alpha), name="alpha")

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

        for options in [bce_options, wass_options]:
            options.update(dict(warmup_energy=warmup_energy))

        # Losses definition
        self._mae_loss = TF_MAE()
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
        # MAE loss
        output = transformer((source, target), training=training)
        energy_mask = tf.cast(
            target[:, :, 2] >= self._warmup_energy, dtype=target.dtype
        )
        if sample_weight is None:
            sample_weight = tf.identity(energy_mask)
        else:
            sample_weight *= energy_mask
        mae_loss = self._mae_loss(target, output, sample_weight=sample_weight)

        # Adversarial loss
        adv_loss = self._adv_loss.transformer_loss(
            transformer=transformer,
            discriminator=discriminator,
            source=source,
            target=target,
            sample_weight=sample_weight,
            training=training,
        )

        tot_loss = mae_loss + self._alpha * adv_loss
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
    def warmup_energy(self) -> float:
        return self._warmup_energy

    @property
    def init_alpha(self) -> float:
        return self._init_alpha

    @property
    def alpha(self) -> tf.Variable:
        return self._alpha

    @property
    def adversarial_metric(self) -> str:
        return self._adversarial_metric

    @property
    def bce_options(self) -> dict:
        return self._bce_options

    @property
    def wass_options(self) -> dict:
        return self._wass_options
