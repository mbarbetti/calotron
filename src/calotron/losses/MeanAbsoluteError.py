import tensorflow as tf
from tensorflow import keras

from calotron.losses.AdvLoss import AdvLoss


class MeanAbsoluteError(AdvLoss):
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
        warmup_energy=1e-8,
        name="mae_loss",
    ) -> None:
        super().__init__(
            alpha=alpha,
            adversarial_metric=adversarial_metric,
            bce_options=bce_options,
            wass_options=wass_options,
            warmup_energy=warmup_energy,
            name=name,
        )

        # MAE loss definition
        self._mae_loss = keras.losses.MeanAbsoluteError(
            reduction=tf.keras.losses.Reduction.NONE
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

        if sample_weight is None:
            sample_weight = tf.ones(shape=tf.shape(target)[:2])
        mask = tf.cast(sample_weight > 0.0, dtype=target.dtype)

        energy_mask = tf.cast(
            target[:, :, 2] >= self._warmup_energy, dtype=target.dtype
        )
        mask *= energy_mask

        # MAE loss
        mae_loss = self._mae_loss(target, output)
        mae_loss *= mask
        mae_loss = tf.reduce_sum(sample_weight * mae_loss) / tf.reduce_sum(
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
            main_loss=mae_loss, adv_loss=adv_loss, alpha=self._alpha
        )
