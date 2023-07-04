import tensorflow as tf

from calotron.layers import Encoder, MappingNet, SynthesisNet
from calotron.models.transformers.Transformer import (
    START_TOKEN_INITIALIZERS,
    Transformer,
)


class GigaGenerator(Transformer):
    def __init__(
        self,
        output_depth,
        encoder_depth,
        mapping_latent_dim,
        synthesis_depth,
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
        super(Transformer, self).__init__(name=name, dtype=dtype)

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

        # MappingNet
        self._map_net = MappingNet(
            output_dim=synthesis_depth,
            latent_dim=mapping_latent_dim,
            num_layers=num_layers,
            hidden_units=mlp_units,
            dropout_rate=dropout_rate,
            output_activation=None,
            name="mapping_net",
            dtype=self.dtype,
        )

        # SynthesisNet
        self._synth_net = SynthesisNet(
            output_depth=synthesis_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            key_dim=key_dim,
            mlp_units=mlp_units,
            dropout_rate=dropout_rate,
            seq_ord_latent_dim=seq_ord_latent_dim,
            seq_ord_max_length=seq_ord_max_length,
            seq_ord_normalization=seq_ord_normalization,
            enable_res_smoothing=enable_res_smoothing,
            name="synthesis_net",
            dtype=self.dtype,
        )

        # Final layers
        self._output_layer, self._multi_activations = self._prepare_final_layers(
            output_depth=self._output_depth,
            output_activations=self._output_activations,
            dtype=self.dtype,
        )

    def call(self, inputs) -> tf.Tensor:
        source, target = inputs
        target = self._prepare_input_target(target)
        enc_out = self._encoder(source)
        map_out = self._map_net(enc_out[:, -1, :])
        synth_out = self._synth_net(target, w=map_out, condition=enc_out[:, :-1, :])
        out = self._output_layer(synth_out)
        if self._multi_activations is not None:
            out = self._multi_activations(out)
        return out

    @property
    def mapping_output_dim(self) -> int:
        return self._map_net.output_dim

    @property
    def synthesis_output_depth(self) -> int:
        return self._synth_net.output_depth

    @property
    def mapping_latent_dim(self) -> int:
        return self._map_net.latent_dim

    @property
    def mapping_num_layers(self) -> int:
        return self._map_net.num_layers

    @property
    def synthesis_num_layers(self) -> int:
        return self._synth_net.num_layers

    @property
    def synthesis_num_heads(self) -> int:
        return self._synth_net.num_heads

    @property
    def synthesis_key_dim(self) -> int:
        return self._synth_net.key_dim

    @property
    def mapping_mlp_units(self) -> int:
        return self._map_net.hidden_units

    @property
    def synthesis_mlp_units(self) -> int:
        return self._synth_net.mlp_units

    @property
    def mapping_dropout_rate(self) -> float:
        return self._map_net.dropout_rate

    @property
    def synthesis_dropout_rate(self) -> float:
        return self._synth_net.dropout_rate

    @property
    def synthesis_seq_ord_latent_dim(self) -> int:
        return self._synth_net.seq_ord_latent_dim

    @property
    def synthesis_seq_ord_max_length(self) -> int:
        return self._synth_net.seq_ord_max_length

    @property
    def synthesis_seq_ord_normalization(self) -> float:
        return self._synth_net.seq_ord_normalization

    @property
    def enable_res_smoothing(self) -> bool:
        return self._encoder.enable_res_smoothing

    @property
    def attention_weights(self) -> tf.Tensor:
        return self._synth_net._last_attn_scores
