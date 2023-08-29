import tensorflow as tf
from tensorflow import keras

from calotron.layers import SeqOrderEmbedding
from calotron.models.discriminators import Discriminator
from calotron.models.players import Decoder, Encoder, PretrainedEncoder


class GigaDiscriminator(Discriminator):
    def __init__(
        self,
        output_units,
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
        output_activation=None,
        pretrained_encoder_dir=None,
        additional_encoder_layers=None,
        name=None,
        dtype=None,
    ) -> None:
        super(Discriminator, self).__init__(name=name, dtype=dtype)
        self._condition_aware = True

        # Output units
        assert isinstance(output_units, (int, float))
        assert output_units >= 1
        self._output_units = int(output_units)

        # Output activation
        self._output_activation = output_activation

        # Encoder
        if pretrained_encoder_dir is not None:
            self._encoder = PretrainedEncoder(
                output_depth=encoder_depth,
                num_layers=additional_encoder_layers
                if additional_encoder_layers
                else num_layers,
                num_heads=num_heads,
                key_dim=key_dim,
                admin_res_scale=admin_res_scale,
                mlp_units=mlp_units,
                dropout_rate=dropout_rate,
                seq_ord_latent_dim=seq_ord_latent_dim,
                seq_ord_max_length=seq_ord_max_length,
                seq_ord_normalization=seq_ord_normalization,
                enable_res_smoothing=enable_res_smoothing,
                pretrained_model_dir=pretrained_encoder_dir,
                name="pretrain_encoder",
                dtype=self.dtype,
            )
        else:
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

        # Sequence order embedding
        self._seq_ord_embed = SeqOrderEmbedding(
            latent_dim=encoder_depth,
            max_length=seq_ord_max_length,
            normalization=seq_ord_normalization,
            dropout_rate=dropout_rate,
            name="seq_ord_embed",
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
            autoregressive_mode=False,
            name="decoder",
            dtype=self.dtype,
        )

        # Final layers
        self._avg_pool = keras.layers.GlobalAveragePooling1D(name="avg_pool")
        self._max_pool = keras.layers.GlobalMaxPooling1D(name="max_pool")
        self._concat = keras.layers.Concatenate(name="concat")
        self._seq = self._prepare_final_layers(
            output_units=self._output_units,
            latent_dim=2 * decoder_depth,
            num_layers=3,
            min_units=2 * self._output_units,
            dropout_rate=dropout_rate,
            output_activation=self._output_activation,
            dtype=self.dtype,
        )

    def call(self, inputs, padding_mask=None) -> tf.Tensor:
        source, target = inputs
        if padding_mask is not None:
            padding_mask = tf.tile(
                padding_mask[:, :, None], (1, 1, tf.shape(target)[2])
            )
            target *= padding_mask
        enc_out = self._encoder(source)
        enc_out = self._seq_ord_embed(enc_out)
        dec_out = self._decoder((target, enc_out))
        out_avg = self._avg_pool(dec_out)
        out_max = self._max_pool(dec_out)
        out = self._concat([out_avg, out_max])
        for layer in self._seq:
            out = layer(out)
        return out

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
    def output_activation(self):  # TODO: add Union[str, Activation]
        return self._output_activation

    @property
    def pretrained_encoder_dir(self):  # TODO: add Union[str, None]
        return self._encoder.pretrained_model_dir

    @property
    def additional_encoder_layers(self) -> int:
        return self._encoder.num_layers

    @property
    def attention_weights(self) -> tf.Tensor:
        return self._decoder._last_attn_scores
