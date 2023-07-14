import tensorflow as tf

from calotron.losses.AdvLoss import AdvLoss


class ConservationLaw(AdvLoss):
    def __init__(
        self,
        alpha=1.0,
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
        name="conserv_law_loss",
    ) -> None:
        super().__init__(
            alpha=alpha,
            adversarial_metric=adversarial_metric,
            bce_options=bce_options,
            wass_options=wass_options,
            warmup_energy=warmup_energy,
            name=name,
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

        return self._compute_adv_loss(
            main_loss=l2_loss,  # conserv_loss,  # monotonic_loss,
            adv_loss=adv_loss,
            alpha=self._alpha,
        )
