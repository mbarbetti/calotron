import tensorflow as tf

from calotron.losses.MeanSquaredError import MeanSquaredError as MSE


class GeomReinfMSE(MSE):
    def __init__(
        self,
        rho=0.1,
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
        warmup_energy=1e-8,
        name="geom_reinf_mse_loss",
    ) -> None:
        super().__init__(
            alpha=alpha,
            adversarial_metric=adversarial_metric,
            bce_options=bce_options,
            wass_options=wass_options,
            warmup_energy=warmup_energy,
            name=name,
        )

        # Geometric patience
        assert isinstance(rho, (int, float))
        assert rho > 0.0
        self._rho = float(rho)

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

        if sample_weight is None:
            sample_weight = tf.ones(shape=tf.shape(target)[:2])
        mask = tf.cast(sample_weight > 0.0, dtype=target.dtype)

        energy_mask = tf.cast(
            target[:, :, 2] >= self._warmup_energy, dtype=target.dtype
        )
        mask *= energy_mask

        # Geometric reinforcement
        distance = tf.math.sqrt(
            tf.reduce_sum((target[:, :, :2] - output[:, :, :2]) ** 2, axis=-1)
        )
        geom_matches = tf.cast(distance < self._rho, dtype=target.dtype)
        geom_matches *= mask

        # Geometric loss
        num_target_points = tf.reduce_sum(mask, axis=-1)
        num_geom_matches = tf.reduce_sum(geom_matches, axis=-1)
        geom_loss = tf.reduce_mean(
            tf.abs(num_target_points - num_geom_matches) / num_target_points
        )

        # MSE loss
        mse_loss = self._mse_loss(target, output)
        mse_loss *= mask
        mse_loss = tf.reduce_sum(sample_weight * mse_loss) / tf.reduce_sum(
            sample_weight
        )

        # Adversarial loss
        adv_loss = self._adv_loss.transformer_loss(
            transformer=transformer,
            discriminator=discriminator,
            source=source,
            target=target,
            sample_weight=sample_weight,
            training=training,
        )

        return self._compute_mixed_loss(
            main_loss=geom_loss + mse_loss, adv_loss=adv_loss, alpha=self._alpha
        )

    @property
    def rho(self) -> float:
        return self._rho
