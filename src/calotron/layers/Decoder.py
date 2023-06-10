import tensorflow as tf

from calotron.layers.Attention import CausalSelfAttention, CrossAttention
from calotron.layers.MultilayerPerceptron import MultilayerPerceptron
from calotron.layers.SeqOrderEmbedding import SeqOrderEmbedding


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        output_depth,
        num_heads,
        key_dim,
        num_res_layers,
        admin_res_scale="O(n)",
        mlp_units=128,
        dropout_rate=0.1,
        name=None,
        dtype=None,
    ) -> None:
        super().__init__(name=name, dtype=dtype)
        if name is not None:
            prefix = name.split("_")[0]
            suffix = name.split("_")[-1]

        # Masked multi-head attention
        self._causal_attn = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            embed_dim=output_depth,
            num_res_layers=num_res_layers,
            admin_res_scale=admin_res_scale,
            dropout_rate=dropout_rate,
            name=f"{prefix}_causal_attn_{suffix}" if name else None,
            dtype=self.dtype,
        )

        # Multi-head attention
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
        self, x, condition, causal_attn_mask=None, cross_attn_mask=None
    ) -> tf.Tensor:
        f_x = self._causal_attn(x, attention_mask=causal_attn_mask)
        f_x = self._cross_attn(f_x, condition, attention_mask=cross_attn_mask)
        self._attn_scores = self._cross_attn._attn_scores
        out = self._mlp(f_x)
        return out

    @property
    def output_depth(self) -> int:
        return self._mlp.output_units

    @property
    def num_heads(self) -> int:
        return self._causal_attn.num_heads

    @property
    def key_dim(self) -> int:
        return self._causal_attn.key_dim

    @property
    def num_res_layers(self) -> int:
        return self._causal_attn.num_res_layers

    @property
    def admin_res_scale(self) -> str:
        return self._causal_attn.admin_res_scale

    @property
    def mlp_units(self) -> int:
        return self._mlp.hidden_units

    @property
    def dropout_rate(self) -> float:
        return self._causal_attn.dropout_rate


class Decoder(tf.keras.layers.Layer):
    def __init__(
        self,
        output_depth,
        num_layers,
        num_heads,
        key_dim,
        admin_res_scale="O(n)",
        mlp_units=128,
        dropout_rate=0.1,
        seq_ord_latent_dim=16,
        seq_ord_max_length=512,
        seq_ord_normalization=10_000,
        enable_residual_smoothing=True,
        name=None,
        dtype=None,
    ) -> None:
        super().__init__(name=name, dtype=dtype)

        # Number of layers
        assert isinstance(num_layers, (int, float))
        assert num_layers >= 1
        self._num_layers = int(num_layers)

        # Residual smoothing
        assert isinstance(enable_residual_smoothing, bool)
        self._enable_residual_smoothing = enable_residual_smoothing

        # Sequence order embedding
        self._seq_ord_embedding = SeqOrderEmbedding(
            latent_dim=seq_ord_latent_dim,
            max_length=seq_ord_max_length,
            normalization=seq_ord_normalization,
            dropout_rate=dropout_rate,
            name="dec_so_embedding",
            dtype=self.dtype,
        )

        # Smoothing layer
        if self._enable_residual_smoothing:
            self._smooth_layer = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(
                        units=output_depth,
                        activation="relu",
                        kernel_initializer="glorot_normal",
                        bias_initializer="zeros",
                        name="dec_sl_dense",
                        dtype=self.dtype,
                    ),
                    tf.keras.layers.Dropout(
                        dropout_rate, name="dec_sl_dropout", dtype=self.dtype
                    ),
                ]
            )
        else:
            self._smooth_layer = None

        # Decoder layers
        self._decoder_layers = [
            DecoderLayer(
                output_depth=output_depth,
                num_heads=num_heads,
                key_dim=key_dim,
                num_res_layers=2 * self._num_layers,
                admin_res_scale=admin_res_scale,
                mlp_units=mlp_units,
                dropout_rate=dropout_rate,
                name=f"dec_layer_{i}",
                dtype=self.dtype,
            )
            for i in range(self._num_layers)
        ]
        self._last_attn_scores = None

    def call(
        self, x, condition, causal_attn_mask=None, cross_attn_mask=None
    ) -> tf.Tensor:
        out = self._seq_ord_embedding(x)
        if self._smooth_layer is not None:
            out = self._smooth_layer(out)
        for i in range(self._num_layers):
            out = self._decoder_layers[i](
                out, condition, causal_attn_mask, cross_attn_mask
            )
        self._last_attn_scores = self._decoder_layers[-1]._attn_scores
        return out

    @property
    def output_depth(self) -> int:
        return self._decoder_layers[0].output_depth

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_heads(self) -> int:
        return self._decoder_layers[0].num_heads

    @property
    def key_dim(self) -> int:
        return self._decoder_layers[0].key_dim

    @property
    def num_res_layers(self) -> int:
        return self._decoder_layers[0].num_res_layers

    @property
    def admin_res_scale(self) -> str:
        return self._decoder_layers[0].admin_res_scale

    @property
    def mlp_units(self) -> int:
        return self._decoder_layers[0].mlp_units

    @property
    def dropout_rate(self) -> float:
        return self._seq_ord_embedding.dropout_rate

    @property
    def seq_ord_latent_dim(self) -> int:
        return self._seq_ord_embedding.latent_dim

    @property
    def seq_ord_max_length(self) -> int:
        return self._seq_ord_embedding.max_length

    @property
    def seq_ord_normalization(self) -> float:
        return self._seq_ord_embedding.normalization

    @property
    def enable_residual_smoothing(self) -> bool:
        return self._enable_residual_smoothing
