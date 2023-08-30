import tensorflow as tf
from tensorflow import keras

from calotron.losses.BaseLoss import BaseLoss


class KLDivergence(BaseLoss):
    def __init__(self, warmup_energy=1e-8, name="kl_loss") -> None:
        super().__init__(name)

        # Warmup energy
        assert isinstance(warmup_energy, (int, float))
        assert warmup_energy >= 0.0
        self._warmup_energy = float(warmup_energy)

        # TensorFlow KLDivergence
        self._loss = keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)

    def transformer_loss(
        self,
        transformer,
        discriminator,
        source,
        target,
        sample_weight=None,
        training=True,
    ) -> tf.Tensor:
        y_true, y_pred, evt_weights, _ = self._perform_classification(
            source=source,
            target=target,
            transformer=transformer,
            discriminator=discriminator,
            warmup_energy=self._warmup_energy,
            inj_noise_std=0.0,
            sample_weight=sample_weight,
            training_transformer=training,
            training_discriminator=False,
            return_transformer_output=False,
        )

        kl_loss = self._loss(y_true, y_pred)
        kl_loss = tf.reduce_sum(evt_weights * kl_loss) / tf.reduce_sum(evt_weights)
        return kl_loss  # divergence minimization

    def discriminator_loss(
        self,
        transformer,
        discriminator,
        source,
        target,
        sample_weight=None,
        training=True,
    ) -> tf.Tensor:
        y_true, y_pred, evt_weights, _ = self._perform_classification(
            source=source,
            target=target,
            transformer=transformer,
            discriminator=discriminator,
            warmup_energy=self._warmup_energy,
            inj_noise_std=0.0,
            sample_weight=sample_weight,
            training_transformer=False,
            training_discriminator=training,
            return_transformer_output=False,
        )

        kl_loss = self._loss(y_true, y_pred)
        kl_loss = tf.reduce_sum(evt_weights * kl_loss) / tf.reduce_sum(evt_weights)
        return -kl_loss  # divergence maximization

    @property
    def warmup_energy(self) -> float:
        return self._warmup_energy
