import tensorflow as tf


class BaseLoss:
    def __init__(self, name="loss") -> None:
        assert isinstance(name, str)
        self._name = name

    @staticmethod
    def _perform_classification(
        source,
        target,
        transformer,
        discriminator,
        warmup_energy=0.0,
        inj_noise_std=0.0,
        sample_weight=None,
        training_transformer=False,
        training_discriminator=False,
        return_transformer_output=False,
    ) -> tuple:
        output = transformer((source, target), training=training_transformer)

        if sample_weight is None:
            sample_weight = tf.ones(shape=tf.shape(target)[:2])
        mask = tf.cast(sample_weight > 0.0, dtype=target.dtype)
        evt_weights = tf.reduce_mean(sample_weight, axis=-1)

        energy_mask = tf.cast(target[:, :, 2] >= warmup_energy, dtype=target.dtype)
        mask *= energy_mask

        source_concat = tf.concat([source, source], axis=0)
        target_concat = tf.concat([target, output], axis=0)
        mask_concat = tf.concat([mask, mask], axis=0)

        if inj_noise_std > 0.0:
            rnd_noise = tf.random.normal(
                tf.shape(target_concat), stddev=inj_noise_std, dtype=target.dtype
            )
        else:
            rnd_noise = 0.0

        d_out = discriminator(
            (
                source_concat,
                tf.clip_by_value(
                    target_concat + rnd_noise,
                    clip_value_min=tf.reduce_min(target_concat, axis=[0, 1]),
                    clip_value_max=tf.reduce_max(target_concat, axis=[0, 1]),
                ),
            ),
            padding_mask=mask_concat,
            training=training_discriminator,
        )
        y_true, y_pred = tf.split(d_out, 2, axis=0)

        if return_transformer_output:
            return y_true, y_pred, evt_weights, mask, output
        else:
            return y_true, y_pred, evt_weights, mask

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
            "Only `BaseLoss` subclasses have the "
            "`transformer_loss()` method implemented."
        )

    def discriminator_loss(
        self,
        transformer,
        discriminator,
        source,
        target,
        sample_weight=None,
        training=True,
    ) -> tf.Tensor:
        raise NotImplementedError(
            "Only `BaseLoss` subclasses have the "
            "`discriminator_loss()` method implemented."
        )

    @property
    def name(self) -> str:
        return self._name
