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

        # Latent space layers
        self._latent_layers = list()
        for _ in range(self._num_layers - 1):
            self._latent_layers.append(
                tf.keras.layers.Dense(
                    self._hidden_units, activation="relu", dtype=self.dtype
                )
            )
            self._latent_layers.append(
                tf.keras.layers.Dropout(self._dropout_rate, dtype=self.dtype)
            )
        self._latent_layers += [
            tf.keras.layers.Dense(self._latent_dim, activation="relu", dtype=self.dtype)
        ]

        # Output layers
        self._output_layers = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    self._latent_dim, activation="relu", dtype=self.dtype
                ),
                tf.keras.layers.Dropout(self._dropout_rate, dtype=self.dtype),
                tf.keras.layers.Dense(self._latent_dim, dtype=self.dtype),
            ]
        )

    def call(self, x) -> tf.Tensor:
        outputs = list()
        for i in range(x.shape[1]):
            latent_tensor = x[:, i : i + 1, :]
            for layer in self._latent_layers:
                latent_tensor = layer(latent_tensor)
            outputs.append(latent_tensor)

        concat = tf.keras.layers.Concatenate(axis=1)(outputs)
        output = tf.reduce_sum(concat, axis=1)
        output = self._output_layers(output)
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
