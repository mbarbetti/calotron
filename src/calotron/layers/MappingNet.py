import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Layer, LeakyReLU

LEAKY_ALPHA = 0.1
SEED = 42


class MappingNet(Layer):
    def __init__(
        self,
        output_dim,
        latent_dim,
        num_layers,
        hidden_units=128,
        dropout_rate=0.0,
        output_activation=None,
        name=None,
        dtype=None,
    ) -> None:
        super().__init__(name=name, dtype=dtype)

        # Output dimension
        assert output_dim >= 1
        self._output_dim = int(output_dim)

        # Latent space dimension
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

        # Output activation
        self._output_activation = output_activation

        # MappingNet layers
        self._seq = list()
        for i in range(self._num_layers - 1):
            self._seq.append(
                Dense(
                    units=self._hidden_units,
                    activation=None,
                    kernel_initializer="glorot_uniform",
                    bias_initializer="zeros",
                    name=f"map_dense_{i}" if name else None,
                    dtype=self.dtype,
                )
            )
            self._seq.append(
                LeakyReLU(
                    alpha=LEAKY_ALPHA, name=f"map_leaky_relu_{i}" if name else None
                )
            )
            self._seq.append(
                Dropout(
                    rate=self._dropout_rate, name=f"map_dropout_{i}" if name else None
                )
            )
        self._seq.append(
            Dense(
                units=output_dim,
                activation=output_activation,
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
                name="map_dense_out" if name else None,
                dtype=self.dtype,
            )
        )

    def call(self, x) -> tf.Tensor:
        x = self._prepare_input(x, seed=None)
        for layer in self._seq:
            x = layer(x)
        return x

    def generate(self, x, seed=None) -> tf.Tensor:
        tf.random.set_seed(seed=SEED)
        x = self._prepare_input(x, seed=seed)
        for layer in self._seq:
            x = layer(x)
        return x

    def _prepare_input(self, x, seed=None) -> tf.Tensor:
        latent_sample = tf.random.normal(
            shape=(tf.shape(x)[0], self._latent_dim),
            mean=0.0,
            stddev=1.0,
            dtype=self.dtype,
            seed=seed,
        )
        x = tf.concat([x, latent_sample], axis=-1)
        return x

    @property
    def output_dim(self) -> int:
        return self._output_dim

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

    @property
    def output_activation(self):
        return self._output_activation
