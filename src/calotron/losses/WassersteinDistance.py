import tensorflow as tf

from calotron.losses.BaseLoss import BaseLoss

LIPSCHITZ_REGULARIZERS = ["gp", "alp"]
PENALTY_STRATEGIES = ["two-sided", "one-sided"]
VIRTUAL_DIR_UPDS = 1
FIXED_XI = 1.0
SAMPLED_XI_MIN = 0.8
SAMPLED_XI_MAX = 1.2
EPSILON = 1e-12
LIPSCHITZ_CONSTANT = 1.0


class WassersteinDistance(BaseLoss):
    def __init__(
        self,
        lipschitz_regularizer="alp",
        lipschitz_penalty=100.0,
        lipschitz_penalty_strategy="one-sided",
        warmup_energy=0.0,
        name="wass_dist_loss",
    ) -> None:
        super().__init__(name)

        # Warmup energy
        assert isinstance(warmup_energy, (int, float))
        assert warmup_energy >= 0.0
        self._warmup_energy = float(warmup_energy)

        # Lipschitz regularizer
        assert isinstance(lipschitz_regularizer, str)
        if lipschitz_regularizer not in LIPSCHITZ_REGULARIZERS:
            raise ValueError(
                "`lipschitz_regularizer` should be selected "
                f"in {PENALTY_STRATEGIES}, instead "
                f"'{lipschitz_regularizer}' passed"
            )
        self._lipschitz_regularizer = lipschitz_regularizer

        # Lipschitz penalty
        assert isinstance(lipschitz_penalty, (int, float))
        assert lipschitz_penalty > 0.0
        self._lipschitz_penalty = float(lipschitz_penalty)

        # Penalty strategy
        assert isinstance(lipschitz_penalty_strategy, str)
        if lipschitz_penalty_strategy not in PENALTY_STRATEGIES:
            raise ValueError(
                "`lipschitz_penalty_strategy` should be selected "
                f"in {PENALTY_STRATEGIES}, instead "
                f"'{lipschitz_penalty_strategy}' passed"
            )
        self._lipschitz_penalty_strategy = lipschitz_penalty_strategy

    def transformer_loss(
        self,
        transformer,
        discriminator,
        source,
        target,
        sample_weight=None,
        training=True,
    ) -> tf.Tensor:
        trainset_true, trainset_pred = self._prepare_adv_trainset(
            source=source,
            target=target,
            transformer=transformer,
            warmup_energy=self._warmup_energy,
            sample_weight=sample_weight,
            training_transformer=training,
        )
        source_true, target_true, evt_w_true, mask_true = trainset_true
        source_pred, target_pred, evt_w_pred, mask_pred = trainset_pred

        source_concat = tf.concat([source_true, source_pred], axis=0)
        target_concat = tf.concat([target_true, target_pred], axis=0)
        mask_concat = tf.concat([mask_true, mask_pred], axis=0)
        d_out = discriminator(
            (source_concat, target_concat), padding_mask=mask_concat, training=False
        )
        y_true, y_pred = tf.split(d_out, 2, axis=0)

        # Real target loss
        real_loss = tf.reduce_sum(evt_w_true * y_true) / tf.reduce_sum(evt_w_true)
        real_loss = tf.cast(real_loss, dtype=target_true.dtype)

        # Fake target loss
        fake_loss = tf.reduce_sum(evt_w_pred * y_pred) / tf.reduce_sum(evt_w_pred)
        fake_loss = tf.cast(fake_loss, dtype=target_pred.dtype)

        return tf.stop_gradient(real_loss) - fake_loss

    def discriminator_loss(
        self,
        transformer,
        discriminator,
        source,
        target,
        sample_weight=None,
        training=True,
    ) -> tf.Tensor:
        trainset_true, trainset_pred = self._prepare_adv_trainset(
            source=source,
            target=target,
            transformer=transformer,
            warmup_energy=self._warmup_energy,
            sample_weight=sample_weight,
            training_transformer=False,
        )
        source_true, target_true, evt_w_true, mask_true = trainset_true
        source_pred, target_pred, evt_w_pred, mask_pred = trainset_pred

        source_concat = tf.concat([source_true, source_pred], axis=0)
        target_concat = tf.concat([target_true, target_pred], axis=0)
        mask_concat = tf.concat([mask_true, mask_pred], axis=0)
        d_out = discriminator(
            (source_concat, target_concat), padding_mask=mask_concat, training=training
        )
        y_true, y_pred = tf.split(d_out, 2, axis=0)

        # Real target loss
        real_loss = tf.reduce_sum(evt_w_true * y_true) / tf.reduce_sum(evt_w_true)
        real_loss = tf.cast(real_loss, dtype=target_true.dtype)

        # Fake target loss
        fake_loss = tf.reduce_sum(evt_w_pred * y_pred) / tf.reduce_sum(evt_w_pred)
        fake_loss = tf.cast(fake_loss, dtype=target_pred.dtype)

        if discriminator.condition_aware:
            shuffled_source = tf.random.shuffle(source_true)
            masked_target = target_true * tf.tile(
                mask_true[:, :, None], (1, 1, tf.shape(target_true)[2])
            )
            y_pred = discriminator(
                (shuffled_source, masked_target), padding_mask=None, training=training
            )

            # Fake source loss
            source_loss = tf.reduce_mean(y_pred)
            source_loss = tf.cast(source_loss, dtype=target_pred.dtype)

            loss = ((fake_loss - real_loss) + (source_loss - real_loss)) / 2.0
        else:
            loss = fake_loss - real_loss

        reg = self._lipschitz_regularization(
            sample_true=(source_true, target_true, mask_true, y_true),
            sample_pred=(source_pred, target_pred, mask_pred, y_pred),
            discriminator=discriminator,
            training_discriminator=training,
        )
        return loss + reg

    def _lipschitz_regularization(
        self, sample_true, sample_pred, discriminator, training_discriminator=True
    ) -> tf.Tensor:
        source_true, target_true, mask_true, y_true = sample_true
        source_pred, target_pred, mask_pred, y_pred = sample_pred

        if self._lipschitz_regularizer == "gp":
            target_concat = tf.concat([target_true, target_pred], axis=0)

            with tf.GradientTape() as tape:
                # Compute interpolated points
                eps = tf.tile(
                    tf.random.uniform(
                        shape=(tf.shape(target_true)[0], tf.shape(target_true)[1]),
                        minval=0.0,
                        maxval=1.0,
                        dtype=target_true.dtype,
                    )[:, :, None],
                    (1, 1, tf.shape(target_true)[2]),
                )
                target_hat = tf.clip_by_value(
                    target_pred + eps * (target_true - target_pred),
                    clip_value_min=tf.reduce_min(target_concat, axis=[0, 1]),
                    clip_value_max=tf.reduce_max(target_concat, axis=[0, 1]),
                )
                tape.watch(target_hat)

                # Value of the discriminator on interpolated points
                y_hat = discriminator(
                    (source_pred, target_hat),
                    padding_mask=mask_pred,
                    training=training_discriminator,
                )
                grad = tape.gradient(y_hat, target_hat) + EPSILON  # non-zero gradient
                norm = tf.norm(grad, axis=-1)

            if self._lipschitz_penalty_strategy == "two-sided":
                gp_term = (norm - LIPSCHITZ_CONSTANT) ** 2
            else:
                gp_term = (tf.maximum(0.0, norm - LIPSCHITZ_CONSTANT)) ** 2
            return self._lipschitz_penalty * tf.reduce_mean(gp_term)

        else:
            source_concat = tf.concat([source_true, source_pred], axis=0)
            target_concat = tf.concat([target_true, target_pred], axis=0)
            mask_concat = tf.concat([mask_true, mask_pred], axis=0)
            d_out = tf.concat([y_true, y_pred], axis=0)

            # Initial virtual adversarial direction
            adv_dir = tf.random.uniform(
                shape=(
                    2 * tf.shape(target_true)[0],
                    tf.shape(target_true)[1],
                    tf.shape(target_true)[2],
                ),
                minval=-1.0,
                maxval=1.0,
                dtype=target_true.dtype,
            )
            adv_dir /= tf.norm(adv_dir, axis=[1, 2], keepdims=True)

            for _ in range(VIRTUAL_DIR_UPDS):
                with tf.GradientTape() as tape:
                    tape.watch(adv_dir)
                    target_hat = tf.clip_by_value(
                        target_concat + FIXED_XI * adv_dir,
                        clip_value_min=tf.reduce_min(target_concat, axis=[0, 1]),
                        clip_value_max=tf.reduce_max(target_concat, axis=[0, 1]),
                    )
                    d_hat = discriminator(
                        (source_concat, target_hat),
                        padding_mask=mask_concat,
                        training=training_discriminator,
                    )
                    y_diff = tf.reduce_mean(tf.abs(d_out - d_hat))
                    grad = tape.gradient(y_diff, adv_dir) + EPSILON  # non-zero gradient
                    adv_dir = grad / tf.norm(grad, axis=[1, 2], keepdims=True)

            # Virtual adversarial direction
            xi = tf.random.uniform(
                shape=(2 * tf.shape(target_true)[0],),
                minval=SAMPLED_XI_MIN,
                maxval=SAMPLED_XI_MAX,
                dtype=target_true.dtype,
            )
            xi = tf.tile(
                xi[:, None, None],
                (1, tf.shape(target_true)[1], tf.shape(target_true)[2]),
            )
            target_hat = tf.clip_by_value(
                target_concat + xi * adv_dir,
                clip_value_min=tf.reduce_min(target_concat, axis=[0, 1]),
                clip_value_max=tf.reduce_max(target_concat, axis=[0, 1]),
            )
            d_hat = discriminator(
                (source_concat, target_hat),
                padding_mask=mask_concat,
                training=training_discriminator,
            )
            y_diff = tf.abs(d_out - d_hat)
            x_diff = tf.norm(
                tf.abs(target_concat - target_hat) + EPSILON,  # non-zero difference
                axis=[1, 2],
                keepdims=True,
            )

            K = y_diff / x_diff  # lipschitz constant
            if self._lipschitz_penalty_strategy == "two-sided":
                alp_term = tf.abs(K - LIPSCHITZ_CONSTANT)
            else:
                alp_term = tf.maximum(0.0, K - LIPSCHITZ_CONSTANT)
            return self._lipschitz_penalty * tf.reduce_mean(alp_term) ** 2

    @property
    def warmup_energy(self) -> float:
        return self._warmup_energy

    @property
    def lipschitz_regularizer(self) -> str:
        return self._lipschitz_regularizer

    @property
    def lipschitz_penalty(self) -> float:
        return self._lipschitz_penalty

    @property
    def lipschitz_penalty_strategy(self) -> str:
        return self._lipschitz_penalty_strategy
