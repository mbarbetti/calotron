import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout

from calotron.layers import DeepSets


class Discriminator(tf.keras.Model):
    def __init__(
        self,
        output_units,
        latent_dim=64,
        deepsets_dense_num_layers=4,
        deepsets_dense_units=128,
        dropout_rate=0.0,
        output_activation=None,
        name=None,
        dtype=None,
    ) -> None:
        super().__init__(name=name, dtype=dtype)
        self._condition_aware = False

        # Output units
        assert isinstance(output_units, (int, float))
        assert output_units >= 1
        self._output_units = int(output_units)

        # Output activation
        self._output_activation = output_activation

        # Deep Sets
        self._deep_sets = DeepSets(
            latent_dim=latent_dim,
            dense_num_layers=deepsets_dense_num_layers,
            dense_units=deepsets_dense_units,
            dropout_rate=dropout_rate,
            name="deepsets",
            dtype=self.dtype,
        )

        # Final layers
        self._seq = self._prepare_final_layers(
            output_units=self._output_units,
            latent_dim=latent_dim,
            num_layers=3,
            min_units=4,
            dropout_rate=dropout_rate,
            output_activation=self._output_activation,
            dtype=self.dtype,
        )

    @staticmethod
    def _prepare_final_layers(
        output_units,
        latent_dim,
        num_layers=3,
        min_units=4,
        dropout_rate=0.0,
        output_activation=None,
        dtype=None,
    ) -> list:
        final_layers = list()
        for i in range(num_layers - 1):
            seq_units = max(latent_dim / (i + 1.0), min_units)
            final_layers.append(
                Dense(
                    units=int(seq_units),
                    activation="relu",
                    kernel_initializer="glorot_uniform",
                    bias_initializer="zeros",
                    name=f"dense_{i}",
                    dtype=dtype,
                )
            )
            final_layers.append(
                Dropout(rate=dropout_rate, name=f"dropout_{i}", dtype=dtype)
            )
        final_layers += [
            Dense(
                units=output_units,
                activation=output_activation,
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
                name="dense_out",
                dtype=dtype,
            )
        ]
        return final_layers

    def call(self, inputs, padding_mask=None) -> tf.Tensor:
        _, target = inputs
        out = self._deep_sets(target, padding_mask=padding_mask)
        for layer in self._seq:
            out = layer(out)
        return out

    @property
    def output_units(self) -> int:
        return self._output_units

    @property
    def output_activation(self):  # TODO: add Union[None, activation]
        return self._output_activation

    @property
    def latent_dim(self) -> int:
        return self._deep_sets.latent_dim

    @property
    def deepsets_dense_num_layers(self) -> int:
        return self._deep_sets.dense_num_layers

    @property
    def deepsets_dense_units(self) -> int:
        return self._deep_sets.dense_units

    @property
    def dropout_rate(self) -> float:
        return self._deep_sets.dropout_rate

    @property
    def condition_aware(self) -> bool:
        return self._condition_aware
