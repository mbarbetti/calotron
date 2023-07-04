import tensorflow as tf

from calotron.layers import DeepSets
from calotron.models.discriminators.Discriminator import Discriminator


class PairwiseDiscriminator(Discriminator):
    def __init__(
        self,
        output_units,
        latent_dim=64,
        deepsets_dense_num_layers=1,
        deepsets_dense_units=128,
        deepsets_conv1D_num_layers=3,
        deepsets_conv1D_filters=64,
        deepsets_conv1D_kernel_size=4,
        deepsets_conv1D_strides=4,
        dropout_rate=0.0,
        output_activation=None,
        name=None,
        dtype=None,
    ) -> None:
        super(Discriminator, self).__init__(name=name, dtype=dtype)
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
            conv1D_num_layers=deepsets_conv1D_num_layers,
            conv1D_filters=deepsets_conv1D_filters,
            conv1D_kernel_size=deepsets_conv1D_kernel_size,
            conv1D_strides=deepsets_conv1D_strides,
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

    def call(self, inputs, padding_mask=None) -> tf.Tensor:
        _, target = inputs

        # Pairwise arrangement
        target_1 = tf.tile(target[:, :, None, :], (1, 1, tf.shape(target)[1], 1))
        target_2 = tf.tile(target[:, None, :, :], (1, tf.shape(target)[1], 1, 1))
        pairs = tf.concat([target_1, target_2], axis=-1)
        pairs = tf.reshape(
            pairs,
            shape=(
                tf.shape(target)[0],
                tf.shape(target)[1] ** 2,
                2 * tf.shape(target)[2],
            ),
        )

        # Padding mask arrangement
        if padding_mask is not None:
            mask_1 = tf.tile(
                padding_mask[:, :, None, None], (1, 1, tf.shape(padding_mask)[1], 1)
            )
            mask_2 = tf.tile(
                padding_mask[:, None, :, None], (1, tf.shape(padding_mask)[1], 1, 1)
            )
            mask_pairs = tf.concat([mask_1, mask_2], axis=-1)
            mask_pairs = tf.reshape(
                mask_pairs,
                shape=(tf.shape(padding_mask)[0], tf.shape(padding_mask)[1] ** 2, 2),
            )
            mask_pairs = mask_pairs[:, :, 0] * mask_pairs[:, :, 1]
        else:
            mask_pairs = None

        # Event classification
        out = self._deep_sets(pairs, padding_mask=mask_pairs)
        for layer in self._seq:
            out = layer(out)
        return out

    @property
    def deepsets_conv1D_num_layers(self) -> int:
        return self._deep_sets.conv1D_num_layers

    @property
    def deepsets_conv1D_filters(self) -> int:
        return self._deep_sets.conv1D_filters

    @property
    def deepsets_conv1D_kernel_size(self) -> int:
        return self._deep_sets.conv1D_kernel_size

    @property
    def deepsets_conv1D_strides(self) -> int:
        return self._deep_sets.conv1D_strides
