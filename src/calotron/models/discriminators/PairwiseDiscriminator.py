import tensorflow as tf

from calotron.models.discriminators.Discriminator import Discriminator


class PairwiseDiscriminator(Discriminator):
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
        super().__init__(
            output_units=output_units,
            output_activation=output_activation,
            latent_dim=latent_dim,
            deepsets_num_layers=deepsets_num_layers,
            deepsets_hidden_units=deepsets_hidden_units,
            dropout_rate=dropout_rate,
            name=name,
            dtype=dtype,
        )

    def call(self, inputs, filter=None) -> tf.Tensor:
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

        # Filter arrangement
        if filter is not None:
            filter_1 = tf.tile(filter[:, :, None, None], (1, 1, tf.shape(filter)[1], 1))
            filter_2 = tf.tile(filter[:, None, :, None], (1, tf.shape(filter)[1], 1, 1))
            filter_pairs = tf.concat([filter_1, filter_2], axis=-1)
            filter_pairs = tf.reshape(
                filter_pairs, shape=(tf.shape(filter)[0], tf.shape(filter)[1] ** 2, 2)
            )
            filter_pairs = filter_pairs[:, :, 0] * filter_pairs[:, :, 1]
        else:
            filter_pairs = None

        # Event classification
        output = self._deep_sets(pairs, filter=filter_pairs)
        for layer in self._seq:
            output = layer(output)
        return output
