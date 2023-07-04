import tensorflow as tf
from tensorflow.keras.layers import Dense

from calotron.layers import Decoder, Encoder, MultiActivations

START_TOKEN_INITIALIZERS = ["zeros", "ones", "means"]


class Transformer(tf.keras.Model):
    def __init__(
        self,
        output_depth,
        encoder_depth,
        decoder_depth,
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
        output_activations=None,
        start_token_initializer="ones",
        name=None,
        dtype=None,
    ) -> None:
        super().__init__(name=name, dtype=dtype)

        # Output depth
        assert output_depth >= 1
        self._output_depth = int(output_depth)

        # Output activations
        self._output_activations = output_activations

        # Start token initializer
        assert isinstance(start_token_initializer, str)
        if start_token_initializer not in START_TOKEN_INITIALIZERS:
            raise ValueError(
                "`start_token_initializer` should be selected "
                f"in {START_TOKEN_INITIALIZERS}, instead "
                f"'{start_token_initializer}' passed"
            )
        self._start_token_initializer = start_token_initializer

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

        # Decoder
        self._decoder = Decoder(
            output_depth=decoder_depth,
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
            autoregressive_mode=True,
            name="decoder",
            dtype=self.dtype,
        )

        # Final layers
        self._output_layer, self._multi_activations = self._prepare_final_layers(
            output_depth=self._output_depth,
            output_activations=self._output_activations,
            dtype=self.dtype,
        )

    @staticmethod
    def _prepare_final_layers(
        output_depth, output_activations=None, dtype=None
    ) -> tuple:
        output_layer = Dense(
            units=output_depth,
            activation=None,
            kernel_initializer="he_normal",
            bias_initializer="zeros",
            name="dense_out",
            dtype=dtype,
        )
        if output_activations is not None:
            multi_activations = MultiActivations(
                activations=output_activations,
                output_depth=output_depth,
                name="filter",
                dtype=dtype,
            )
        else:
            multi_activations = None
        return output_layer, multi_activations

    def call(self, inputs) -> tf.Tensor:
        source, target = inputs
        target = self._prepare_input_target(target)
        enc_out = self._encoder(source)
        dec_out = self._decoder(target, condition=enc_out)
        out = self._output_layer(dec_out)
        if self._multi_activations is not None:
            out = self._multi_activations(out)
        return out

    def _prepare_input_target(self, target) -> tf.Tensor:
        if self._start_token_initializer == "zeros":
            start_token = tf.zeros((tf.shape(target)[0], 1, tf.shape(target)[2]))
        elif self._start_token_initializer == "ones":
            zeros = tf.zeros((tf.shape(target)[0], 1, 2))
            ones = tf.ones((tf.shape(target)[0], 1, tf.shape(target)[2] - 2))
            start_token = tf.concat([zeros, ones], axis=-1)
        elif self._start_token_initializer == "means":
            start_token = tf.reduce_mean(target, axis=(0, 1))[None, None, :]
            start_token = tf.tile(start_token, (tf.shape(target)[0], 1, 1))
        return tf.concat([start_token, target[:, :-1, :]], axis=1)

    def get_start_token(self, target) -> tf.Tensor:
        if self._start_token_initializer == "zeros":
            start_token = tf.zeros((tf.shape(target)[0], tf.shape(target)[2]))
        elif self._start_token_initializer == "ones":
            zeros = tf.zeros((tf.shape(target)[0], 2))
            ones = tf.ones((tf.shape(target)[0], tf.shape(target)[2] - 2))
            start_token = tf.concat([zeros, ones], axis=-1)
        elif self._start_token_initializer == "means":
            start_token = tf.reduce_mean(target, axis=(0, 1))[None, :]
            start_token = tf.tile(start_token, (tf.shape(target)[0], 1))
        return start_token

    @property
    def output_depth(self) -> int:
        return self._output_depth

    @property
    def encoder_output_depth(self) -> int:
        return self._encoder.output_depth

    @property
    def decoder_output_depth(self) -> int:
        return self._decoder.output_depth

    @property
    def encoder_num_layers(self) -> int:
        return self._encoder.num_layers

    @property
    def decoder_num_layers(self) -> int:
        return self._decoder.num_layers

    @property
    def encoder_num_heads(self) -> int:
        return self._encoder.num_heads

    @property
    def decoder_num_heads(self) -> int:
        return self._decoder.num_heads

    @property
    def encoder_key_dim(self) -> int:
        return self._encoder.key_dim

    @property
    def decoder_key_dim(self) -> int:
        return self._decoder.key_dim

    @property
    def encoder_admin_res_scale(self) -> str:
        return self._encoder.admin_res_scale

    @property
    def decoder_admin_res_scale(self) -> str:
        return self._decoder.admin_res_scale

    @property
    def encoder_mlp_units(self) -> int:
        return self._encoder.mlp_units

    @property
    def decoder_mlp_units(self) -> int:
        return self._decoder.mlp_units

    @property
    def encoder_dropout_rate(self) -> float:
        return self._encoder.dropout_rate

    @property
    def decoder_dropout_rate(self) -> float:
        return self._decoder.dropout_rate

    @property
    def encoder_seq_ord_latent_dim(self) -> int:
        return self._encoder.seq_ord_latent_dim

    @property
    def decoder_seq_ord_latent_dim(self) -> int:
        return self._decoder.seq_ord_latent_dim

    @property
    def encoder_seq_ord_max_length(self) -> int:
        return self._encoder.seq_ord_max_length

    @property
    def decoder_seq_ord_max_length(self) -> int:
        return self._decoder.seq_ord_max_length

    @property
    def encoder_seq_ord_normalization(self) -> float:
        return self._encoder.seq_ord_normalization

    @property
    def decoder_seq_ord_normalization(self) -> float:
        return self._decoder.seq_ord_normalization

    @property
    def enable_res_smoothing(self) -> bool:
        return self._encoder.enable_res_smoothing

    @property
    def output_activations(self):  # TODO: add Union[list, None]
        return self._output_activations

    @property
    def start_token_initializer(self) -> str:
        return self._start_token_initializer

    @property
    def attention_weights(self) -> tf.Tensor:
        return self._decoder._last_attn_scores
