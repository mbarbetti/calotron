import tensorflow as tf


class DeepSets(tf.keras.layers.Layer):
    def __init__(
        self,
        latent_dim,
        num_layers,
        hidden_units=128,
        dropout_rate=0.1,
        name=None,
        dtype=None,
    ) -> None:
        super().__init__(name=name, dtype=dtype)

        # Latent space dimension
        assert isinstance(latent_dim, (int, float))
        assert latent_dim >= 1
        self._latent_dim = int(latent_dim)

        # Number of layers
        assert isinstance(num_layers, (int, float))
        assert num_layers >= 1
        self._num_layers = int(num_layers)

        # Number of hidden units
        assert isinstance(hidden_units, (int, float))
        assert hidden_units >= 1
        self._hidden_units = int(hidden_units)

        # Dropout rate
        assert isinstance(dropout_rate, (int, float))
        assert dropout_rate >= 0.0 and dropout_rate < 1.0
        self._dropout_rate = float(dropout_rate)

        # Layers
        self._seq = list()
        for _ in range(self._num_layers - 1):
            self._seq.append(
                tf.keras.layers.Dense(
                    self._hidden_units, activation="relu", name="ds_dense", dtype=self.dtype
                )
            )
            self._seq.append(
                tf.keras.layers.Dropout(self._dropout_rate, name="ds_dropout", dtype=self.dtype)
            )
        self._seq += [
            tf.keras.layers.Dense(self._latent_dim, activation="relu", name="ds_output_layer", dtype=self.dtype)
        ]

    def call(self, x) -> tf.Tensor:
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1]
        depth = tf.shape(x)[2]
        output = tf.reshape(x, (batch_size * length, depth))
        for layer in self._seq:
            output = layer(output)
        output = tf.reshape(output, (batch_size, length, self._latent_dim))
        output = tf.reduce_sum(output, axis=1)
        return output

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def num_layers(self) -> int:
        return self._latent_dim

    @property
    def hidden_units(self) -> int:
        return self._hidden_units

    @property
    def dropout_rate(self) -> float:
        return self._dropout_rate
