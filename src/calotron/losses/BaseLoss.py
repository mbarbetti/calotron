import tensorflow as tf


class BaseLoss:
    def __init__(self, name="loss") -> None:
        assert isinstance(name, str)
        self._name = name

    @staticmethod
    def _prepare_clf_trainset(
        source,
        target,
        transformer,
        discriminator,
        warmup_energy=0.0,
        sample_weight=None,
        training_transformer=False,
        training_discriminator=False,
    ) -> tuple:
        output = transformer((source, target), training=training_transformer)

        energy_mask = tf.cast(target[:, :, 2] >= warmup_energy, dtype=target.dtype)
        if sample_weight is None:
            sample_weight = tf.identity(energy_mask)
        else:
            sample_weight *= energy_mask
        evt_weights = tf.reduce_mean(sample_weight, axis=-1)
        mask = tf.cast(sample_weight > 0.0, dtype=target.dtype)

        source_concat = tf.concat([source, source], axis=0)
        target_concat = tf.concat([target, output], axis=0)
        mask_concat = tf.concat([mask, mask], axis=0)

        d_out = discriminator(
            (source_concat, target_concat),
            padding_mask=mask_concat,
            training=training_discriminator,
        )
        y_true, y_pred = tf.split(d_out, 2, axis=0)
        return y_true, y_pred, evt_weights

    @staticmethod
    def _prepare_adv_trainset(
        source,
        target,
        transformer,
        warmup_energy=0.0,
        sample_weight=None,
        training_transformer=False,
    ) -> tuple:
        batch_size = tf.cast(tf.shape(source)[0] / 2, tf.int32)

        source_true, source_pred = tf.split(source[: batch_size * 2], 2, axis=0)
        target_true, target_pred = tf.split(target[: batch_size * 2], 2, axis=0)
        target_pred = transformer(
            (source_pred, target_pred), training=training_transformer
        )

        energy_mask = tf.cast(target[:, :, 2] >= warmup_energy, dtype=target.dtype)
        e_mask_true, e_mask_pred = tf.split(energy_mask[: batch_size * 2], 2, axis=0)

        if sample_weight is None:
            sample_weight = tf.identity(energy_mask)
            w_true, w_pred = tf.split(sample_weight[: batch_size * 2], 2, axis=0)
        else:
            w_true, w_pred = tf.split(sample_weight[: batch_size * 2], 2, axis=0)
            w_true *= e_mask_true
            w_pred *= e_mask_pred
        evt_w_true = tf.reduce_mean(w_true, axis=-1)
        evt_w_pred = tf.reduce_mean(w_pred, axis=-1)

        mask_true = tf.cast(w_true > 0.0, dtype=target.dtype)
        mask_pred = tf.cast(w_pred > 0.0, dtype=target.dtype)

        return (
            (source_true, target_true, evt_w_true, mask_true),
            (source_pred, target_pred, evt_w_pred, mask_pred),
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
