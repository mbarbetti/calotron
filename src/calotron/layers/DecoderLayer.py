import tensorflow as tf
from tensorflow import keras

from calotron.layers.Attention import CrossAttention, SelfAttention
from calotron.layers.MultilayerPerceptron import MultilayerPerceptron


class DecoderLayer(keras.layers.Layer):
    def __init__(
        self,
        output_depth,
        num_heads,
        key_dim,
        num_res_layers,
        admin_res_scale="O(n)",
        mlp_units=128,
        dropout_rate=0.0,
        autoregressive_mode=True,
        name=None,
        dtype=None,
    ) -> None:
        super().__init__(name=name, dtype=dtype)
        if name is not None:
            prefix = name.split("_")[0]
            suffix = name.split("_")[-1]

        # Autoregressive mode
        assert isinstance(autoregressive_mode, bool)
        self._autoregressive_mode = autoregressive_mode

        # Multi-head self-attention
        self._self_attn = SelfAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            embed_dim=output_depth,
            num_res_layers=num_res_layers,
            admin_res_scale=admin_res_scale,
            dropout_rate=dropout_rate,
            name=f"{prefix}_self_attn_{suffix}" if name else None,
            dtype=self.dtype,
        )

        # Multi-head cross-attention
        self._cross_attn = CrossAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            embed_dim=output_depth,
            num_res_layers=num_res_layers,
            admin_res_scale=admin_res_scale,
            dropout_rate=dropout_rate,
            name=f"{prefix}_cross_attn_{suffix}" if name else None,
            dtype=self.dtype,
        )

        # Multilayer perceptron
        self._mlp = MultilayerPerceptron(
            output_units=output_depth,
            hidden_units=mlp_units,
            num_res_layers=num_res_layers,
            admin_res_scale=admin_res_scale,
            dropout_rate=dropout_rate,
            name=f"{prefix}_mlp_{suffix}" if name else None,
            dtype=self.dtype,
        )

    def call(
        self, x, condition, self_attn_mask=None, cross_attn_mask=None
    ) -> tf.Tensor:
        f_x = self._self_attn(
            x, attention_mask=self_attn_mask, use_causal_mask=self._autoregressive_mode
        )
        f_x = self._cross_attn(f_x, condition, attention_mask=cross_attn_mask)
        self._attn_scores = self._cross_attn._attn_scores
        out = self._mlp(f_x)
        return out

    @property
    def output_depth(self) -> int:
        return self._mlp.output_units

    @property
    def num_heads(self) -> int:
        return self._self_attn.num_heads

    @property
    def key_dim(self) -> int:
        return self._self_attn.key_dim

    @property
    def num_res_layers(self) -> int:
        return self._self_attn.num_res_layers

    @property
    def admin_res_scale(self) -> str:
        return self._self_attn.admin_res_scale

    @property
    def mlp_units(self) -> int:
        return self._mlp.hidden_units

    @property
    def dropout_rate(self) -> float:
        return self._self_attn.dropout_rate

    @property
    def autoregressive_mode(self) -> bool:
        return self._autoregressive_mode
