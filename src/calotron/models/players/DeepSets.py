import tensorflow as tf
from tensorflow import keras


class DeepSets(keras.Model):
    def __init__(
        self,
        latent_dim,
        num_layers,
        hidden_units=128,
        dropout_rate=0.0,
        name=None,
        dtype=None,
    ) -> None:
        super().__init__(name=name, dtype=dtype)

        # Latent space dimension
        assert isinstance(latent_dim, (int, float))
        assert latent_dim >= 1
        assert (latent_dim % 2) == 0
        self._latent_dim = int(latent_dim)

        # Number of layers
        assert isinstance(num_layers, (int, float))
        assert num_layers >= 1
        self._num_layers = int(num_layers)

        # Hidden units
        assert isinstance(hidden_units, (int, float))
        assert hidden_units >= 1
        self._hidden_units = int(hidden_units)

        # Dropout rate
        assert isinstance(dropout_rate, (int, float))
        assert dropout_rate >= 0.0 and dropout_rate < 1.0
        self._dropout_rate = float(dropout_rate)

        # Deep Sets layers
        self._seq = list()
        for i in range(self._num_layers - 1):
            self._seq.append(
                keras.layers.Dense(
                    units=self._hidden_units,
                    activation="relu",
                    kernel_initializer="glorot_uniform",
                    bias_initializer="zeros",
                    name=f"dense_{i}" if name else None,
                    dtype=self.dtype,
                )
            )
            self._seq.append(
                keras.layers.Dropout(
                    self._dropout_rate,
                    name=f"dropout_{i}" if name else None,
                    dtype=self.dtype,
                )
            )
        self._seq.append(
            keras.layers.Dense(
                int(self._latent_dim / 2),
                activation=None,
                kernel_initializer="he_normal",
                bias_initializer="zeros",
                name="dense_out" if name else None,
                dtype=self.dtype,
            )
        )

        # Final layers
        self._avg_pool = keras.layers.GlobalAveragePooling1D(
            name="avg_pool" if name else None
        )
        self._max_pool = keras.layers.GlobalMaxPooling1D(
            name="max_pool" if name else None
        )
        self._concat = keras.layers.Concatenate(name="concat" if name else None)

    def call(self, x, padding_mask=None) -> tf.Tensor:
        if padding_mask is not None:
            padding_mask = tf.tile(padding_mask[:, :, None], (1, 1, tf.shape(x)[2]))
            x *= padding_mask
        for layer in self._seq:
            x = layer(x)
        x_avg = self._avg_pool(x)
        x_max = self._max_pool(x)
        out = self._concat([x_avg, x_max])
        return out

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def hidden_units(self) -> int:
        return self._hidden_units

    @property
    def dropout_rate(self) -> float:
        return self._dropout_rate
