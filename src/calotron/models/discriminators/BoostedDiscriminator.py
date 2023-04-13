import tensorflow as tf

from calotron.layers import DeepSets
from calotron.models.discriminators.BaseDiscriminator import BaseDiscriminator

MIN_UNITS = 32.0


class BoostedDiscriminator(BaseDiscriminator):
    def __init__(
        self,
        output_units,
        output_activation=None,
        latent_dim=64,
        deepsets_num_layers=5,
        deepsets_hidden_units=128,
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

        # Number of layers for (both) DeepSets
        assert isinstance(deepsets_num_layers, (int, float))
        assert deepsets_num_layers >= 1
        self._deepsets_num_layers = int(deepsets_num_layers)

        # Number of hidden units for (both) DeepSets
        assert isinstance(deepsets_hidden_units, (int, float))
        assert deepsets_hidden_units >= 1
        self._deepsets_hidden_units = int(deepsets_hidden_units)

        # Dropout rate
        assert isinstance(dropout_rate, (int, float))
        assert dropout_rate >= 0.0 and dropout_rate < 1.0
        self._dropout_rate = float(dropout_rate)

        # Deep Sets for source
        self._deep_sets_source = DeepSets(
            latent_dim=self._latent_dim,
            num_layers=self._deepsets_num_layers,
            hidden_units=self._deepsets_hidden_units,
            dropout_rate=self._dropout_rate,
            name="d_deepsets_source",
            dtype=self.dtype,
        )

        # Deep Sets for target
        self._deep_sets_target = DeepSets(
            latent_dim=self._latent_dim,
            num_layers=self._deepsets_num_layers,
            hidden_units=self._deepsets_hidden_units,
            dropout_rate=self._dropout_rate,
            name="d_deepsets_target",
            dtype=self.dtype,
        )

        # Latent layers
        self._seq = list()
        for i in range(2):
            seq_units = max(2.0 * self._latent_dim / (i + 1.0), MIN_UNITS)
            self._seq.append(
                tf.keras.layers.Dense(
                    units=int(seq_units),
                    activation="relu",
                    kernel_initializer="glorot_uniform",
                    bias_initializer="zeros",
                    # kernel_regularizer="l2",
                    name="d_dense",
                    dtype=self.dtype,
                )
            )
            self._seq.append(
                tf.keras.layers.Dropout(
                    rate=self._dropout_rate, name="d_dropout", dtype=self.dtype
                )
            )

        # Output layer
        self._seq += [
            tf.keras.layers.Dense(
                units=self._output_units,
                activation=self._output_activation,
                kernel_initializer="truncated_normal",
                bias_initializer="zeros",
                # kernel_regularizer="l2",
                name="d_output_layer",
                dtype=self.dtype,
            )
        ]

    def call(self, inputs) -> tf.Tensor:
        source, target = inputs
        output_1 = self._deep_sets_source(source)
        output_2 = self._deep_sets_target(target)
        output = tf.concat([output_1, output_2], axis=-1)
        for layer in self._seq:
            output = layer(output)
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
    def deepsets_num_layers(self) -> int:
        return self._deepsets_num_layers

    @property
    def deepsets_hidden_units(self) -> int:
        return self._deepsets_hidden_units

    @property
    def dropout_rate(self) -> float:
        return self._dropout_rate
