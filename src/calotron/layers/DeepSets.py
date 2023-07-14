import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Layer, LayerNormalization

LN_EPSILON = 0.001


class DeepSets(Layer):
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
                Dense(
                    units=self._hidden_units,
                    activation="relu",
                    kernel_initializer="glorot_uniform",
                    bias_initializer="zeros",
                    name=f"ds_dense_{i}" if name else None,
                    dtype=self.dtype,
                )
            )
            self._seq.append(
                Dropout(
                    self._dropout_rate,
                    name=f"ds_dropout_{i}" if name else None,
                    dtype=self.dtype,
                )
            )
        self._seq.append(
            Dense(
                self._latent_dim,
                activation=None,
                kernel_initializer="he_normal",
                bias_initializer="zeros",
                name="ds_dense_out" if name else None,
                dtype=self.dtype,
            )
        )
        self._evt_ln = LayerNormalization(
            axis=1,
            epsilon=LN_EPSILON,
            name="ds_layer_norm" if name else None,
            dtype=self.dtype,
        )

    def call(self, x, padding_mask=None) -> tf.Tensor:
        if padding_mask is not None:
            padding_mask = tf.tile(padding_mask[:, :, None], (1, 1, tf.shape(x)[2]))
            x *= padding_mask
        for layer in self._seq:
            x = layer(x)
        x = self._evt_ln(x)
        out = tf.reduce_sum(x, axis=1)
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
