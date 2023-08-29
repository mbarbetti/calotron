import tensorflow as tf
from tensorflow import keras

from calotron.models.players import DeepSets


class Discriminator(keras.Model):
    def __init__(
        self,
        output_units,
        latent_dim=64,
        deepsets_num_layers=4,
        deepsets_hidden_units=128,
        dropout_rate=0.0,
        enable_batch_norm=False,
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

        # Batch normalization
        assert isinstance(enable_batch_norm, bool)
        self._enable_batch_norm = enable_batch_norm

        # Output activation
        self._output_activation = output_activation

        # Deep Sets
        self._deep_sets = DeepSets(
            latent_dim=latent_dim,
            num_layers=deepsets_num_layers,
            hidden_units=deepsets_hidden_units,
            dropout_rate=dropout_rate,
            name="deepsets",
            dtype=self.dtype,
        )
        if self._enable_batch_norm:
            self._batch_norm = keras.layers.BatchNormalization(name="batch_norm", dtype=self.dtype)

        # Final layers
        self._seq = self._prepare_final_layers(
            output_units=self._output_units,
            latent_dim=latent_dim,
            num_layers=3,
            min_units=2 * self._output_units,
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
            seq_units = max(latent_dim / (2 * (i + 1)), min_units)
            final_layers.append(
                keras.layers.Dense(
                    units=int(seq_units),
                    activation="relu",
                    kernel_initializer="glorot_uniform",
                    bias_initializer="zeros",
                    name=f"dense_{i}",
                    dtype=dtype,
                )
            )
            final_layers.append(
                keras.layers.Dropout(rate=dropout_rate, name=f"dropout_{i}", dtype=dtype)
            )
        final_layers.append(
            keras.layers.Dense(
                units=output_units,
                activation=output_activation,
                kernel_initializer="he_normal",
                bias_initializer="zeros",
                name="dense_out",
                dtype=dtype,
            )
        )
        return final_layers

    def call(self, inputs, padding_mask=None) -> tf.Tensor:
        _, target = inputs
        out = self._deep_sets(target, padding_mask=padding_mask)
        if self._enable_batch_norm:
            out = self._batch_norm(out)
        for layer in self._seq:
            out = layer(out)
        return out

    @property
    def output_units(self) -> int:
        return self._output_units

    @property
    def latent_dim(self) -> int:
        return self._deep_sets.latent_dim

    @property
    def deepsets_num_layers(self) -> int:
        return self._deep_sets.num_layers

    @property
    def deepsets_hidden_units(self) -> int:
        return self._deep_sets.hidden_units

    @property
    def dropout_rate(self) -> float:
        return self._deep_sets.dropout_rate

    @property
    def enable_batch_norm(self) -> bool:
        return self._enable_batch_norm

    @property
    def output_activation(self):  # TODO: add Union[None, activation]
        return self._output_activation

    @property
    def condition_aware(self) -> bool:
        return self._condition_aware
