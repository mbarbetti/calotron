import tensorflow as tf
from tensorflow.keras.layers import Layer, LayerNormalization


class ModulatedLayerNorm(Layer):
    def __init__(self, axis=-1, epsilon=0.001, name=None, dtype=None) -> None:
        super().__init__(name=name, dtype=dtype)

        # Standard LayerNormalization
        self._ln = LayerNormalization(axis=axis, epsilon=epsilon)

        # Affine transformation parameters
        self._gamma = tf.Variable(
            tf.random.normal(shape=(), mean=0.0, stddev=1.0),
            trainable=True,
            dtype=self.dtype,
        )
        self._beta = tf.Variable(
            tf.random.normal(shape=(), mean=0.0, stddev=1.0),
            trainable=True,
            dtype=self.dtype,
        )

    def call(self, x, w) -> tf.Tensor:
        out = self._ln(x)
        w = tf.tile(w[:, None, :], (1, tf.shape(x)[1], 1))
        return (self._gamma * w + 1.0) * out + self._beta * w

    @property
    def axis(self) -> int:
        return self._ln.axis

    @property
    def epsilon(self) -> float:
        return self._ln.epsilon
