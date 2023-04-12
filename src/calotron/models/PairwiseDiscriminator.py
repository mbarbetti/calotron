import tensorflow as tf

from calotron.models.Discriminator import Discriminator


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

    def call(self, inputs) -> tf.Tensor:
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

        # Event classification
        output = self._deep_sets(pairs)
        for layer in self._seq:
            output = layer(output)
        return output
