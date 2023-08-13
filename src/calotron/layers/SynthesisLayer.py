import tensorflow as tf
from tensorflow.keras.layers import Add, Dense, Dropout, Layer, MultiHeadAttention

from calotron.layers.ModulatedLayerNorm import ModulatedLayerNorm

LN_EPSILON = 0.001
ATTN_DROPOUT_RATE = 0.0


class SynthesisLayer(Layer):
    def __init__(
        self,
        output_depth,
        num_heads,
        key_dim,
        mlp_units=128,
        dropout_rate=0.0,
        name=None,
        dtype=None,
    ) -> None:
        super().__init__(name=name, dtype=dtype)
        if name is not None:
            prefix = name.split("_")[0]
            suffix = name.split("_")[-1]

        # Output depth
        assert isinstance(output_depth, (int, float))
        assert output_depth >= 1
        self._output_depth = int(output_depth)

        # Number of heads
        assert isinstance(num_heads, (int, float))
        assert num_heads >= 1
        self._num_heads = int(num_heads)

        # Key dimension
        assert isinstance(key_dim, (int, float))
        assert key_dim >= 1
        self._key_dim = int(key_dim)

        # MLP hidden units
        assert isinstance(mlp_units, (int, float))
        assert mlp_units >= 1
        self._mlp_units = int(mlp_units)

        # Dropout rate
        assert isinstance(dropout_rate, (int, float))
        assert dropout_rate >= 0.0 and dropout_rate < 1.0
        self._dropout_rate = float(dropout_rate)

        # Modulated layer normalization
        self._ln = ModulatedLayerNorm(
            axis=-1,
            epsilon=LN_EPSILON,
            name=f"{prefix}_ln_{suffix}" if name else None,
            dtype=self.dtype,
        )

        # Addition layer
        self._res = Add(
            name=f"{prefix}_res_{suffix}" if name else None, dtype=self.dtype
        )

        # Multi-head self-attention
        self._self_attn = MultiHeadAttention(
            num_heads=self._num_heads,
            key_dim=self._key_dim,
            value_dim=None,
            dropout=ATTN_DROPOUT_RATE,
            kernel_initializer="he_normal",
            bias_initializer="zeros",
            name=f"{prefix}_self_attn_{suffix}" if name else None,
            dtype=self.dtype,
        )

        # Multi-head cross-attention
        self._cross_attn = MultiHeadAttention(
            num_heads=self._num_heads,
            key_dim=self._key_dim,
            value_dim=None,
            dropout=ATTN_DROPOUT_RATE,
            kernel_initializer="he_normal",
            bias_initializer="zeros",
            name=f"{prefix}_cross_attn_{suffix}" if name else None,
            dtype=self.dtype,
        )

        # Multilayer perceptron layers
        self._mlp = tf.keras.Sequential(
            [
                Dense(
                    units=self._mlp_units,
                    activation="relu",
                    kernel_initializer="he_normal",
                    bias_initializer="zeros",
                    dtype=self.dtype,
                ),
                Dense(
                    units=self._output_depth,
                    activation=None,
                    kernel_initializer="he_normal",
                    bias_initializer="zeros",
                    dtype=self.dtype,
                ),
                Dropout(rate=self._dropout_rate, dtype=self.dtype),
            ],
            name=f"{prefix}_mlp_{suffix}" if name else None,
        )

    def call(self, x, w, condition) -> tf.Tensor:
        # Self attn block
        norm_x = self._ln(x, w)
        f_x = self._self_attn(
            query=norm_x, key=norm_x, value=norm_x, use_causal_mask=True
        )
        x = self._res([x, f_x])

        # Cross attn block
        norm_x = self._ln(x, w)
        f_x, scores = self._cross_attn(
            query=norm_x,
            key=condition,
            value=condition,
            use_causal_mask=False,
            return_attention_scores=True,
        )
        self._attn_scores = scores
        x = self._res([x, f_x])

        # MLP block
        norm_x = self._ln(x, w)
        f_x = self._mlp(norm_x)
        x = self._res([x, f_x])
        return x

    @property
    def output_depth(self) -> int:
        return self._output_depth

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def key_dim(self) -> int:
        return self._key_dim

    @property
    def mlp_units(self) -> int:
        return self._mlp_units

    @property
    def dropout_rate(self) -> float:
        return self._dropout_rate
