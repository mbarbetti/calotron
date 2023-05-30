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

        # Hidden layers
        self._seq = list()
        for _ in range(self._num_layers - 1):
            self._seq.append(
                tf.keras.layers.Dense(
                    self._hidden_units,
                    activation="relu",
                    kernel_initializer="glorot_uniform",
                    bias_initializer="zeros",
                    name="ds_dense",
                    dtype=self.dtype,
                )
            )
            self._seq.append(
                tf.keras.layers.Dropout(
                    self._dropout_rate, name="ds_dropout", dtype=self.dtype
                )
            )

        # Output layer
        self._seq += [
            tf.keras.layers.Dense(
                self._latent_dim,
                activation="relu",
                kernel_initializer="truncated_normal",
                bias_initializer="zeros",
                name="ds_output_layer",
                dtype=self.dtype,
            )
        ]

    def call(self, x, filter=None) -> tf.Tensor:
        for layer in self._seq:
            x = layer(x)
        if filter is not None:
            filter = tf.tile(filter[:, :, None], (1, 1, tf.shape(x)[2]))
            x = tf.math.multiply(x, filter)
        output = tf.reduce_sum(x, axis=1)
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
