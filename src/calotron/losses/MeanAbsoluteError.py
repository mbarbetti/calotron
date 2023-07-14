import tensorflow as tf
from tensorflow.keras.losses import MeanAbsoluteError as TF_MAE

from calotron.losses.AdvLoss import AdvLoss


class MeanAbsoluteError(AdvLoss):
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
        self._mae_loss = TF_MAE()

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
        evt_weights = tf.reduce_mean(sample_weight, axis=-1)

        # MAE loss
        mae_loss = self._mae_loss(target, output, sample_weight=evt_weights)

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
            main_loss=mae_loss, adv_loss=adv_loss, alpha=self._alpha
        )
