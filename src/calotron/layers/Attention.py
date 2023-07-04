import tensorflow as tf
from tensorflow.keras.layers import Layer, LayerNormalization, MultiHeadAttention

from calotron.layers.AdminResidual import AdminResidual

LN_EPSILON = 0.001


class BaseAttention(Layer):
    def __init__(
        self,
        num_heads,
        key_dim,
        embed_dim,
        num_res_layers,
        admin_res_scale="O(n)",
        dropout_rate=0.0,
        name=None,
        dtype=None,
    ) -> None:
        super().__init__(name=name, dtype=dtype)
        if name is not None:
            prefix = "_".join(w for w in name.split("_")[:-1])
            suffix = name.split("_")[-1]
        else:
            prefix, suffix = None, None

        # Number of heads
        assert isinstance(num_heads, (int, float))
        assert num_heads >= 1
        self._num_heads = int(num_heads)

        # Key dimension
        assert isinstance(key_dim, (int, float))
        assert key_dim >= 1
        self._key_dim = int(key_dim)

        # Dropout rate
        assert isinstance(dropout_rate, (int, float))
        assert dropout_rate >= 0.0 and dropout_rate < 1.0
        self._dropout_rate = float(dropout_rate)

        # Attention mechanism layers
        self._mha = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=None,
            dropout=dropout_rate,
            kernel_initializer="he_normal",
            bias_initializer="zeros",
            name=f"{prefix}_mha_{suffix}" if name else None,
            dtype=self.dtype,
        )
        self._res = AdminResidual(
            embed_dim=embed_dim,
            num_res_layers=num_res_layers,
            output_change_scale=admin_res_scale,
            name=f"{prefix}_res_{suffix}" if name else None,
            dtype=self.dtype,
        )
        self._ln = LayerNormalization(
            axis=-1,
            epsilon=LN_EPSILON,
            name=f"{prefix}_ln_{suffix}" if name else None,
            dtype=self.dtype,
        )

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def key_dim(self) -> int:
        return self._key_dim

    @property
    def embed_dim(self) -> int:
        return self._res.embed_dim

    @property
    def num_res_layers(self) -> int:
        return self._res.num_res_layers

    @property
    def admin_res_scale(self) -> str:
        return self._res.output_change_scale

    @property
    def dropout_rate(self) -> float:
        return self._dropout_rate


class CrossAttention(BaseAttention):
    def call(self, x, condition, attention_mask=None) -> tf.Tensor:
        f_x, scores = self._mha(
            query=x,
            key=condition,
            value=condition,
            attention_mask=attention_mask,
            return_attention_scores=True,
        )
        self._attn_scores = scores
        res = self._res([x, f_x])
        out = self._ln(res)
        return out


class SelfAttention(BaseAttention):
    def call(self, x, attention_mask=None, use_causal_mask=False) -> tf.Tensor:
        f_x = self._mha(
            query=x,
            key=x,
            value=x,
            attention_mask=attention_mask,
            use_causal_mask=use_causal_mask,
        )
        res = self._res([x, f_x])
        out = self._ln(res)
        return out
