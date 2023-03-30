import tensorflow as tf

from calotron.losses.BaseLoss import BaseLoss

LIPSCHITZ_CONSTANT = 1.0


class WassersteinDistance(BaseLoss):
    def __init__(
        self,
        lipschitz_penalty=100.0,
        virtual_direction_upds=1,
        xi=10.0,
        epsilon_min=0.01,
        epsilon_max=1.0,
        name="wass_dist_loss",
    ) -> None:
        super().__init__(name)

        # Adversarial Lipschitz penalty
        assert isinstance(lipschitz_penalty, (int, float))
        assert lipschitz_penalty > 0.0
        self._lipschitz_penalty = float(lipschitz_penalty)

        # Virtual adversarial direction updates
        assert isinstance(virtual_direction_upds, (int, float))
        assert virtual_direction_upds > 0
        self._vir_dir_upds = int(virtual_direction_upds)

        # Additional ALP-system hyperparameters
        assert isinstance(xi, (int, float))
        assert xi > 0.0
        self._xi = float(xi)

        assert isinstance(epsilon_min, (int, float))
        assert epsilon_min >= 0.0
        self._epsilon_min = float(epsilon_min)

        assert isinstance(epsilon_max, (int, float))
        assert epsilon_max > epsilon_min
        self._epsilon_max = float(epsilon_max)

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
        y_pred = discriminator(output, training=False)
        if sample_weight is not None:
            sample_weight = tf.reduce_mean(sample_weight, axis=1)
            loss = tf.reduce_sum(sample_weight * y_pred)
            loss /= tf.reduce_sum(sample_weight)
        else:
            loss = tf.reduce_mean(y_pred)
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
        y_true = discriminator(target, training=training)
        y_pred = discriminator(output, training=training)
        if sample_weight is not None:
            sample_weight = tf.reduce_mean(sample_weight, axis=1)
            loss = tf.reduce_sum(sample_weight * (y_true - y_pred))
            loss /= tf.reduce_sum(sample_weight)
        else:
            loss = tf.reduce_mean(y_true - y_pred)

        # Initial virtual adversarial direction
        d = tf.random.uniform(
            tf.shape(target), minval=-0.5, maxval=0.5, dtype=target.dtype
        )
        d /= tf.norm(d, ord="euclidean", axis=[1, 2])[:, None, None]
        with tf.GradientTape() as tape:
            tape.watch(d)
            for _ in range(self._vir_dir_upds):
                target_hat = tf.clip_by_value(
                    target + self._xi * d,
                    clip_value_min=tf.reduce_min(target),
                    clip_value_max=tf.reduce_max(target),
                )
                output_hat = tf.clip_by_value(
                    output + self._xi * d,
                    clip_value_min=tf.reduce_min(output),
                    clip_value_max=tf.reduce_max(output),
                )

                y_true_hat = discriminator(target_hat, training=training)
                y_pred_hat = discriminator(output_hat, training=training)
                y_diff = tf.abs(
                    tf.concat([y_true, y_pred], axis=0)
                    - tf.concat([y_true_hat, y_pred_hat], axis=0)
                )
                y_diff = tf.reduce_mean(y_diff)
            grad = tape.gradient(y_diff, d)
            d = grad / tf.norm(grad, ord="euclidean", axis=[1, 2])[:, None, None]

        # Virtual adversarial direction
        eps = tf.math.exp(
            tf.random.uniform(
                tf.shape(target),
                minval=tf.math.log(self._epsilon_min),
                maxval=tf.math.log(self._epsilon_max),
                dtype=target.dtype,
            )
        )
        target_hat = tf.clip_by_value(
            target + eps * d,
            clip_value_min=tf.reduce_min(target),
            clip_value_max=tf.reduce_max(target),
        )
        output_hat = tf.clip_by_value(
            output + eps * d,
            clip_value_min=tf.reduce_min(output),
            clip_value_max=tf.reduce_max(output),
        )
        x_diff = tf.concat([target, output], axis=0) - tf.concat(
            [target_hat, output_hat], axis=0
        )
        x_diff = tf.maximum(tf.norm(x_diff, ord="euclidean", axis=[1, 2]), 1e-8)

        y_true_hat = discriminator(target_hat, training=training)
        y_pred_hat = discriminator(output_hat, training=training)
        y_diff = tf.abs(
            tf.concat([y_true, y_pred], axis=0)
            - tf.concat([y_true_hat, y_pred_hat], axis=0)
        )

        alp_term = tf.maximum(
            y_diff / x_diff - LIPSCHITZ_CONSTANT, 0.0
        )  # one-side penalty
        alp_term = tf.reduce_mean(alp_term)
        loss += self._lipschitz_penalty * alp_term**2  # adversarial Lipschitz penalty
        loss = tf.cast(loss, dtype=target.dtype)
        return loss

    @property
    def lipschitz_penalty(self) -> float:
        return self._lipschitz_penalty

    @property
    def virtual_direction_upds(self) -> int:
        return self._vir_dir_upds

    @property
    def xi(self) -> float:
        return self._xi

    @property
    def epsilon_min(self) -> float:
        return self._epsilon_min

    @property
    def epsilon_max(self) -> float:
        return self._epsilon_max
