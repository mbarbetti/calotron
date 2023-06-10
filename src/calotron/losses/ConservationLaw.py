import tensorflow as tf

from calotron.losses.BaseLoss import BaseLoss
from calotron.losses.BinaryCrossentropy import BinaryCrossentropy
from calotron.losses.WassersteinDistance import WassersteinDistance

ADV_METRICS = ["binary-crossentropy", "wasserstein-distance"]


class ConservationLaw(BaseLoss):
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
        name="conserv_law_loss",
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

        for options in [self._bce_options, self._wass_options]:
            options.update(dict(warmup_energy=warmup_energy))

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
        output = transformer((source, target), training=training)
        energy_mask = tf.cast(
            target[:, :, 2] >= self._warmup_energy, dtype=target.dtype
        )

        # L2 loss
        l2_norm = tf.math.sqrt(tf.reduce_sum((target - output) ** 2, axis=-1))
        if sample_weight is None:
            sample_weight = tf.identity(energy_mask)
        else:
            sample_weight *= energy_mask
        evt_errors = tf.reduce_sum(l2_norm * sample_weight, axis=-1)
        batch_rmse = tf.math.sqrt(tf.reduce_mean(evt_errors**2))
        l2_loss = batch_rmse / tf.cast(tf.shape(target)[1], dtype=target.dtype)
        l2_loss = tf.cast(l2_loss, dtype=target.dtype)

        # Energy conservation
        filter = tf.cast(sample_weight > 0.0, dtype=target.dtype)
        target_tot_energy = tf.reduce_sum(target[:, :, 2] * filter, axis=-1)
        output_tot_energy = tf.reduce_sum(output[:, :, 2] * filter, axis=-1)
        batch_rmse = tf.math.sqrt(
            tf.reduce_mean((target_tot_energy - output_tot_energy) ** 2)
        )
        conserv_loss = batch_rmse / tf.cast(tf.shape(target)[1], dtype=target.dtype)
        conserv_loss = tf.cast(conserv_loss, dtype=target.dtype)

        # Monotonic function
        # output_energy = tf.math.maximum(output[:, :, 2], 1e-8)
        # non_monotonic_func = output_energy[:, 1:] / output_energy[:, :-1] > 1.0
        # count_non_monotonic = tf.cast(
        #     tf.math.count_nonzero(non_monotonic_func, axis=-1), dtype=target.dtype
        # )
        # monotonic_loss = tf.reduce_mean(count_non_monotonic) / tf.cast(
        #     tf.shape(target)[1], dtype=target.dtype
        # )
        # monotonic_loss = tf.cast(monotonic_loss, dtype=target.dtype)

        # Adversarial loss
        adv_loss = self._adv_loss.transformer_loss(
            transformer=transformer,
            discriminator=discriminator,
            source=source,
            target=target,
            sample_weight=sample_weight,
            training=training,
        )

        tot_loss = l2_loss + self._alpha * adv_loss
        # tot_loss = l2_loss + conserv_loss + self._alpha * adv_loss
        # tot_loss = l2_loss + conserv_loss + monotonic_loss + self._alpha * adv_loss
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
