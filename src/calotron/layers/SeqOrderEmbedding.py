import numpy as np
import tensorflow as tf


class SeqOrderEmbedding(tf.keras.layers.Layer):
    def __init__(
        self, latent_dim=16, max_length=512, normalization=10_000, name=None, dtype=None
    ) -> None:
        super().__init__(name=name, dtype=dtype)
        if name is not None:
            prefix = name.split("_")[0]

        # Sequence order latent space dimension
        assert isinstance(latent_dim, (int, float))
        assert latent_dim >= 1
        self._latent_dim = int(latent_dim)

        # Sequence max length
        assert isinstance(max_length, (int, float))
        assert max_length >= 1
        self._max_length = int(max_length)

        # Sequence order encoding normalization
        assert isinstance(normalization, (int, float))
        assert normalization > 0.0
        self._normalization = float(normalization)

        # Sequence order encoding
        self._seq_ord_encoding = self._seq_order_encoding(
            length=self._max_length,
            depth=self._latent_dim,
            normalization=self._normalization,
            dtype=self.dtype,
        )

        # Embedding layer
        self._embedding = tf.keras.layers.Dense(
            units=self._latent_dim,
            activation="linear",
            kernel_initializer="glorot_uniform",
            name=f"{prefix}_seq_ord_dense" if name else None,
            dtype=self.dtype,
        )

    def call(self, x) -> tf.Tensor:
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1]
        seq_order = tf.tile(
            self._seq_ord_encoding[None, :length, :], (batch_size, 1, 1)
        )
        output = tf.concat([x, seq_order], axis=2)
        output = self._embedding(output)
        output /= tf.norm(output, axis=2, keepdims=True)  # normalized arrays
        return output

    @staticmethod
    def _seq_order_encoding(
        length, depth, normalization=10_000, dtype=tf.float32
    ) -> tf.Tensor:
        pos_encoding = np.zeros(shape=(length, depth))  # buffer to fill
        for k in range(length):
            for i in range(int(depth / 2)):
                denominator = np.power(normalization, 2 * i / depth)
                pos_encoding[k, 2 * i] = np.sin(k / denominator)
                pos_encoding[k, 2 * i + 1] = np.cos(k / denominator)
        return tf.cast(pos_encoding, dtype=dtype)

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def normalization(self) -> float:
        return self._normalization
