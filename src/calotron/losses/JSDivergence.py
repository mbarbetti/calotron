import tensorflow as tf
from tensorflow.keras.losses import KLDivergence as TF_KLDivergence

from calotron.losses.BaseLoss import BaseLoss


class JSDivergence(BaseLoss):
    def __init__(self, warmup_energy=0.0, name="js_loss") -> None:
        super().__init__(name)

        # Warmup energy
        assert isinstance(warmup_energy, (int, float))
        assert warmup_energy >= 0.0
        self._warmup_energy = float(warmup_energy)

        # TensorFlow KLDivergence
        self._kl_div = TF_KLDivergence()

    def transformer_loss(
        self,
        transformer,
        discriminator,
        source,
        target,
        sample_weight=None,
        training=True,
    ) -> tf.Tensor:
        y_true, y_pred, evt_weights = self._prepare_clf_trainset(
            source=source,
            target=target,
            transformer=transformer,
            discriminator=discriminator,
            warmup_energy=self._warmup_energy,
            sample_weight=sample_weight,
            training_transformer=training,
            training_discriminator=False,
        )

        loss = self._js_div(y_true, y_pred, sample_weight=evt_weights)
        loss = tf.cast(loss, dtype=target.dtype)
        return loss  # divergence minimization

    def discriminator_loss(
        self,
        transformer,
        discriminator,
        source,
        target,
        sample_weight=None,
        training=True,
    ) -> tf.Tensor:
        y_true, y_pred, evt_weights = self._prepare_clf_trainset(
            source=source,
            target=target,
            transformer=transformer,
            discriminator=discriminator,
            warmup_energy=self._warmup_energy,
            sample_weight=sample_weight,
            training_transformer=False,
            training_discriminator=training,
        )

        loss = self._js_div(y_true, y_pred, sample_weight=evt_weights)
        loss = tf.cast(loss, dtype=target.dtype)
        return -loss  # divergence maximization

    def _js_div(self, y_true, y_pred, sample_weight=None) -> tf.Tensor:
        loss = 0.5 * self._kl_div(
            y_true, 0.5 * (y_true + y_pred), sample_weight=sample_weight
        ) + 0.5 * self._kl_div(
            y_pred, 0.5 * (y_true + y_pred), sample_weight=sample_weight
        )
        return loss

    @property
    def warmup_energy(self) -> float:
        return self._warmup_energy
