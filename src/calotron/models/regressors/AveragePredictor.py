import tensorflow as tf
from tensorflow import keras

from calotron.models.players import Encoder


class AveragePredictor(keras.Model):
    def __init__(
        self,
        output_units,
        encoder_depth,
        num_layers,
        num_heads,
        key_dim,
        admin_res_scale="O(n)",
        mlp_units=128,
        dropout_rate=0.0,
        seq_ord_latent_dim=16,
        seq_ord_max_length=512,
        seq_ord_normalization=10_000,
        enable_res_smoothing=True,
        output_activation=None,
        name=None,
        dtype=None,
    ) -> None:
        super().__init__(name=name, dtype=dtype)

        # Output units
        assert output_units >= 1
        self._output_units = int(output_units)

        # Output activation
        self._output_activation = output_activation

        # Encoder
        self._encoder = Encoder(
            output_depth=encoder_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            key_dim=key_dim,
            admin_res_scale=admin_res_scale,
            mlp_units=mlp_units,
            dropout_rate=dropout_rate,
            seq_ord_latent_dim=seq_ord_latent_dim,
            seq_ord_max_length=seq_ord_max_length,
            seq_ord_normalization=seq_ord_normalization,
            enable_res_smoothing=enable_res_smoothing,
            name="encoder",
            dtype=self.dtype,
        )

        # Final layers
        self._avg_pool = keras.layers.GlobalAveragePooling1D(name="avg_pool")
        self._max_pool = keras.layers.GlobalMaxPooling1D(name="max_pool")
        self._concat = keras.layers.Concatenate(name="concat")
        self._seq = self._prepare_final_layers(
            output_units=self._output_units,
            latent_dim=2 * encoder_depth,
            num_layers=3,
            min_units=2 * self._output_units,
            dropout_rate=dropout_rate,
            output_activation=output_activation,
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
                keras.layers.Dropout(
                    rate=dropout_rate, name=f"dropout_{i}", dtype=dtype
                )
            )
        final_layers.append(
            keras.layers.Dense(
                units=output_units,
                activation=output_activation,
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
                name="dense_out",
                dtype=dtype,
            )
        )
        return final_layers

    def call(self, x) -> tf.Tensor:
        enc_out = self._encoder(x)
        out_avg = self._avg_pool(enc_out)
        out_max = self._max_pool(enc_out)
        out = self._concat([out_avg, out_max])
        for layer in self._seq:
            out = layer(out)
        return out

    @property
    def output_units(self) -> int:
        return self._output_units

    @property
    def encoder_depth(self) -> int:
        return self._encoder.output_depth

    @property
    def num_layers(self) -> int:
        return self._encoder.num_layers

    @property
    def num_heads(self) -> int:
        return self._encoder.num_heads

    @property
    def key_dim(self) -> int:
        return self._encoder.key_dim

    @property
    def admin_res_scale(self) -> str:
        return self._encoder.admin_res_scale

    @property
    def mlp_units(self) -> int:
        return self._encoder.mlp_units

    @property
    def dropout_rate(self) -> float:
        return self._encoder.dropout_rate

    @property
    def seq_ord_latent_dim(self) -> int:
        return self._encoder.seq_ord_latent_dim

    @property
    def seq_ord_max_length(self) -> int:
        return self._encoder.seq_ord_max_length

    @property
    def seq_ord_normalization(self) -> float:
        return self._encoder.seq_ord_normalization

    @property
    def enable_res_smoothing(self) -> bool:
        return self._encoder.enable_res_smoothing

    @property
    def output_activation(self):  # TODO: add Union[None, activation]
        return self._output_activation

    @property
    def encoder(self) -> Encoder:
        return self._encoder
