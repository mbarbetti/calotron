import tensorflow as tf
from tensorflow.keras.losses import KLDivergence as TF_KLDivergence

from calotron.losses.BaseLoss import BaseLoss


class KLDivergence(BaseLoss):
    def __init__(self, warmup_energy=0.0, name="kl_loss") -> None:
        super().__init__(name)

        # Warmup energy
        assert isinstance(warmup_energy, (int, float))
        assert warmup_energy >= 0.0
        self._warmup_energy = float(warmup_energy)

        # TensorFlow KLDivergence
        self._loss = TF_KLDivergence()

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
        mask = tf.cast(sample_weight > 0.0, dtype=target.dtype)

        y_true = discriminator((source, target), padding_mask=mask, training=False)
        y_pred = discriminator((source, output), padding_mask=mask, training=False)

        loss = self._loss(y_true, y_pred)
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
        output = transformer((source, target), training=False)
        energy_mask = tf.cast(
            target[:, :, 2] >= self._warmup_energy, dtype=target.dtype
        )
        if sample_weight is None:
            sample_weight = tf.identity(energy_mask)
        else:
            sample_weight *= energy_mask
        mask = tf.cast(sample_weight > 0.0, dtype=target.dtype)

        y_true = discriminator((source, target), padding_mask=mask, training=training)
        y_pred = discriminator((source, output), padding_mask=mask, training=training)

        loss = self._loss(y_true, y_pred)
        loss = tf.cast(loss, dtype=target.dtype)
        return -loss  # divergence maximization

    @property
    def warmup_energy(self) -> float:
        return self._warmup_energy
