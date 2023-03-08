import tensorflow as tf

from calotron.layers import DeepSets


class Discriminator(tf.keras.Model):
    def __init__(
        self,
        output_units,
        output_activation=None,
        latent_dim=64,
        hidden_layers=5,
        hidden_units=128,
        dropout_rate=0.1,
        name=None,
        dtype=None,
    ) -> None:
        super().__init__(name=name, dtype=dtype)

        # Output units
        assert isinstance(output_units, (int, float))
        assert output_units >= 1
        self._output_units = int(output_units)

        # Output activation
        self._output_activation = output_activation

        # Latent space dimension
        assert isinstance(latent_dim, (int, float))
        assert latent_dim >= 1
        self._latent_dim = int(latent_dim)

        # Number of hidden layers
        assert isinstance(hidden_layers, (int, float))
        assert hidden_layers >= 1
        self._hidden_layers = int(hidden_layers)

        # Number of hidden units
        assert isinstance(hidden_units, (int, float))
        assert hidden_units >= 1
        self._hidden_units = int(hidden_units)

        # Dropout rate
        assert isinstance(dropout_rate, (int, float))
        assert dropout_rate >= 0.0 and dropout_rate < 1.0
        self._dropout_rate = float(dropout_rate)

        # Deep Sets
        self._deepsets = DeepSets(
            latent_dim=self._latent_dim,
            num_layers=self._hidden_layers,
            hidden_units=self._hidden_units,
            dropout_rate=self._dropout_rate,
            dtype=self.dtype,
        )

        # Output layers
        self._dropout = tf.keras.layers.Dropout(self._dropout_rate, dtype=self.dtype)
        self._output_layer = tf.keras.layers.Dense(
            self._output_units, name="output_layer", dtype=self.dtype
        )

    def call(self, x) -> tf.Tensor:
        x = self._deepsets(x)
        output = self._dropout(x)
        output = self._output_layer(output)
        return output

    @property
    def output_units(self) -> int:
        return self._output_units

    @property
    def output_activation(self):  # TODO: add Union[None, activation]
        return self._output_activation

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def hidden_layers(self) -> int:
        return self._hidden_layers

    @property
    def hidden_units(self) -> int:
        return self._hidden_units

    @property
    def dropout_rate(self) -> float:
        return self._dropout_rate

    @property
    def deepsets(self) -> DeepSets:
        return self._deepsets
