import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy as TF_BCE

from calotron.losses.BaseLoss import BaseLoss


class BinaryCrossentropy(BaseLoss):
    def __init__(
        self,
        injected_noise_stddev=0.0,
        from_logits=False,
        label_smoothing=0.0,
        name="bce_loss",
    ) -> None:
        super().__init__(name)

        # Noise standard deviation
        assert isinstance(injected_noise_stddev, (int, float))
        assert injected_noise_stddev >= 0.0
        self._inj_noise_std = float(injected_noise_stddev)

        # BCE `from_logits` flag
        assert isinstance(from_logits, bool)
        self._from_logits = from_logits

        # BCE `label_smoothing`
        assert isinstance(label_smoothing, (int, float))
        assert label_smoothing >= 0.0 and label_smoothing <= 1.0
        self._label_smoothing = float(label_smoothing)

        # TensorFlow BinaryCrossentropy
        self._loss = TF_BCE(
            from_logits=self._from_logits, label_smoothing=self._label_smoothing
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
        if sample_weight is not None:
            evt_weights = tf.reduce_mean(sample_weight, axis=1)
        else:
            evt_weights = None

        # Adversarial loss
        output = transformer((source, target), training=training)
        if self._inj_noise_std > 0.0:
            rnd_pred = tf.random.normal(
                tf.shape(output), stddev=self._inj_noise_std, dtype=output.dtype
            )
        else:
            rnd_pred = 0.0
        y_pred = discriminator((source, output + rnd_pred), training=False)
        adv_loss = self._loss(tf.ones_like(y_pred), y_pred, sample_weight=evt_weights)
        adv_loss = tf.cast(adv_loss, dtype=output.dtype)
        return adv_loss

    def discriminator_loss(
        self,
        transformer,
        discriminator,
        source,
        target,
        sample_weight=None,
        training=True,
    ) -> tf.Tensor:
        if sample_weight is not None:
            evt_weights = tf.reduce_mean(sample_weight, axis=1)
        else:
            evt_weights = None

        # Real target loss
        if self._inj_noise_std > 0.0:
            rnd_true = tf.random.normal(
                tf.shape(target), stddev=self._inj_noise_std, dtype=target.dtype
            )
        else:
            rnd_true = 0.0
        y_true = discriminator((source, target + rnd_true), training=training)
        real_loss = self._loss(tf.ones_like(y_true), y_true, sample_weight=evt_weights)
        real_loss = tf.cast(real_loss, dtype=target.dtype)

        # Fake target loss
        output = transformer((source, target), training=False)
        if self._inj_noise_std > 0.0:
            rnd_pred = tf.random.normal(
                tf.shape(output), stddev=self._inj_noise_std, dtype=output.dtype
            )
        else:
            rnd_pred = 0.0
        y_pred = discriminator((source, output + rnd_pred), training=training)
        fake_loss = self._loss(tf.zeros_like(y_pred), y_pred, sample_weight=evt_weights)
        fake_loss = tf.cast(fake_loss, dtype=output.dtype)
        return (real_loss + fake_loss) / 2.0

    @property
    def injected_noise_stddev(self) -> float:
        return self._inj_noise_std

    @property
    def from_logits(self) -> bool:
        return self._from_logits

    @property
    def label_smoothing(self) -> float:
        return self._label_smoothing
