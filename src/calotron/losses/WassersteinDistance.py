import tensorflow as tf

from calotron.losses.BaseLoss import BaseLoss

LIPSCHITZ_CONSTANT = 1.0


class WassersteinDistance(BaseLoss):
    def __init__(
        self,
        warmup_energy=0.0,
        lipschitz_penalty=100.0,
        virtual_direction_upds=1,
        fixed_xi=10.0,
        sampled_xi_min=0.0,
        sampled_xi_max=1.0,
        epsilon=1e-12,
        name="wass_dist_loss",
    ) -> None:
        super().__init__(name)

        # Warmup energy
        assert isinstance(warmup_energy, (int, float))
        assert warmup_energy >= 0.0
        self._warmup_energy = float(warmup_energy)

        # Adversarial Lipschitz penalty
        assert isinstance(lipschitz_penalty, (int, float))
        assert lipschitz_penalty > 0.0
        self._lipschitz_penalty = float(lipschitz_penalty)

        # Virtual adversarial direction updates
        assert isinstance(virtual_direction_upds, (int, float))
        assert virtual_direction_upds > 0
        self._vir_dir_upds = int(virtual_direction_upds)

        # Additional ALP-system hyperparameters
        assert isinstance(fixed_xi, (int, float))
        assert fixed_xi > 0.0
        self._fixed_xi = float(fixed_xi)

        assert isinstance(sampled_xi_min, (int, float))
        assert sampled_xi_min >= 0.0
        self._sampled_xi_min = float(sampled_xi_min)

        assert isinstance(sampled_xi_max, (int, float))
        assert sampled_xi_max > sampled_xi_min
        self._sampled_xi_max = float(sampled_xi_max)

        assert isinstance(epsilon, (int, float))
        assert epsilon > 0.0
        self._epsilon = float(epsilon)

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
        if sample_weight is None:
            sample_weight = tf.identity(energy_mask)
        else:
            sample_weight *= energy_mask

        y_true = discriminator((source, target), filter=sample_weight, training=False)
        y_pred = discriminator((source, output), filter=sample_weight, training=False)

        loss = tf.reduce_mean(y_pred - y_true)
        loss = tf.cast(loss, dtype=output.dtype)
        return loss

    def discriminator_loss(
        self,
        transformer,
        discriminator,
        source,
        target,
        sample_weight=None,
        training=True,
    ) -> tf.Tensor:
        output = transformer((source, target), training=False)
        energy_mask = tf.cast(
            target[:, :, 2] >= self._warmup_energy, dtype=target.dtype
        )
        if sample_weight is None:
            sample_weight = tf.identity(energy_mask)
        else:
            sample_weight *= energy_mask

        y_true = discriminator(
            (source, target), filter=sample_weight, training=training
        )
        y_pred = discriminator(
            (source, output), filter=sample_weight, training=training
        )

        loss = tf.reduce_mean(y_true - y_pred)

        # Initial virtual adversarial direction
        batch_size = tf.shape(target)[0]
        length = tf.shape(target)[1]
        depth = tf.shape(target)[2]
        with tf.GradientTape() as tape:
            d = tf.random.uniform(
                shape=(2 * batch_size, length, depth),
                minval=-1.0,
                maxval=1.0,
                dtype=target.dtype,
            )
            d /= tf.norm(d, axis=[1, 2])[:, None, None]
            tape.watch(d)
            for _ in range(self._vir_dir_upds):
                d_target, d_output = tf.split(d, 2, axis=0)
                target_hat = tf.clip_by_value(
                    target + self._fixed_xi * d_target,
                    clip_value_min=tf.reduce_min(target),
                    clip_value_max=tf.reduce_max(target),
                )
                output_hat = tf.clip_by_value(
                    output + self._fixed_xi * d_output,
                    clip_value_min=tf.reduce_min(output),
                    clip_value_max=tf.reduce_max(output),
                )
                y_true_hat = discriminator((source, target_hat), training=training)
                y_pred_hat = discriminator((source, output_hat), training=training)
                y_diff = tf.abs(
                    tf.concat([y_true, y_pred], axis=0)
                    - tf.concat([y_true_hat, y_pred_hat], axis=0)
                )
                y_diff = tf.reduce_mean(y_diff)
                grad = tape.gradient(y_diff, d) + self._epsilon  # non-zero gradient
                d = grad / tf.norm(grad, axis=[1, 2], keepdims=True)

        # Virtual adversarial direction
        xi = tf.random.uniform(
            shape=(2 * batch_size, length, depth),
            minval=self._sampled_xi_min,
            maxval=self._sampled_xi_max,
            dtype=target.dtype,
        )
        xi_target, xi_output = tf.split(xi, 2, axis=0)
        d_target, d_output = tf.split(d, 2, axis=0)
        target_hat = tf.clip_by_value(
            target + xi_target * d_target,
            clip_value_min=tf.reduce_min(target),
            clip_value_max=tf.reduce_max(target),
        )
        output_hat = tf.clip_by_value(
            output + xi_output * d_output,
            clip_value_min=tf.reduce_min(output),
            clip_value_max=tf.reduce_max(output),
        )
        x_diff = (
            tf.abs(
                tf.concat([target, output], axis=0)
                - tf.concat([target_hat, output_hat], axis=0)
            )
            + self._epsilon
        )  # non-zero difference
        x_diff = tf.norm(x_diff, axis=[1, 2], keepdims=True)

        y_true_hat = discriminator((source, target_hat), training=training)
        y_pred_hat = discriminator((source, output_hat), training=training)
        y_diff = tf.abs(
            tf.concat([y_true, y_pred], axis=0)
            - tf.concat([y_true_hat, y_pred_hat], axis=0)
        )

        K = y_diff / x_diff
        alp_term = tf.maximum(K - LIPSCHITZ_CONSTANT, 0.0)  # one-side penalty
        alp_term = tf.reduce_mean(alp_term)
        loss += self._lipschitz_penalty * alp_term**2  # adversarial Lipschitz penalty
        loss = tf.cast(loss, dtype=target.dtype)
        return loss

    @property
    def warmup_energy(self) -> float:
        return self._warmup_energy

    @property
    def lipschitz_penalty(self) -> float:
        return self._lipschitz_penalty

    @property
    def virtual_direction_upds(self) -> int:
        return self._vir_dir_upds

    @property
    def fixed_xi(self) -> float:
        return self._fixed_xi

    @property
    def sampled_xi_min(self) -> float:
        return self._sampled_xi_min

    @property
    def sampled_xi_max(self) -> float:
        return self._sampled_xi_max

    @property
    def epsilon(self) -> float:
        return self._epsilon
