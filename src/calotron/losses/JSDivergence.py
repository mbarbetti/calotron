import tensorflow as tf
from tensorflow import keras

from calotron.losses.BaseLoss import BaseLoss


class JSDivergence(BaseLoss):
    def __init__(self, warmup_energy=1e-8, name="js_loss") -> None:
        super().__init__(name)

        # Warmup energy
        assert isinstance(warmup_energy, (int, float))
        assert warmup_energy >= 0.0
        self._warmup_energy = float(warmup_energy)

        # TensorFlow KLDivergence
        self._kl_div = keras.losses.KLDivergence(
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

        js_loss = self._js_div(y_true, y_pred, sample_weight=evt_weights)
        return js_loss  # divergence minimization

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

        js_loss = self._js_div(y_true, y_pred, sample_weight=evt_weights)
        return -js_loss  # divergence maximization

    def _js_div(self, y_true, y_pred, sample_weight=None) -> tf.Tensor:
        y_interp = 0.5 * (y_true + y_pred)
        js_loss = 0.5 * (
            self._kl_div(y_true, y_interp) + self._kl_div(y_pred, y_interp)
        )
        if sample_weight is not None:
            js_loss = tf.reduce_sum(sample_weight * js_loss) / tf.reduce_sum(
                sample_weight
            )
        else:
            js_loss = tf.reduce_mean(js_loss)
        return js_loss

    @property
    def warmup_energy(self) -> float:
        return self._warmup_energy
