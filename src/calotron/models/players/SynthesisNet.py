import tensorflow as tf
from tensorflow import keras

from calotron.layers import SeqOrderEmbedding, SynthesisLayer

LN_EPSILON = 0.001
ATTN_DROPOUT_RATE = 0.0


class SynthesisNet(keras.Model):
    def __init__(
        self,
        output_depth,
        num_layers,
        num_heads,
        key_dim,
        mlp_units=128,
        dropout_rate=0.0,
        seq_ord_latent_dim=16,
        seq_ord_max_length=512,
        seq_ord_normalization=10_000,
        enable_res_smoothing=True,
        name=None,
        dtype=None,
    ) -> None:
        super().__init__(name=name, dtype=dtype)

        # Number of layers
        assert isinstance(num_layers, (int, float))
        assert num_layers >= 1
        self._num_layers = int(num_layers)

        # Residual smoothing
        assert isinstance(enable_res_smoothing, bool)
        self._enable_res_smoothing = enable_res_smoothing

        # Sequence order embedding
        self._seq_ord_embed = SeqOrderEmbedding(
            latent_dim=seq_ord_latent_dim,
            max_length=seq_ord_max_length,
            normalization=seq_ord_normalization,
            dropout_rate=dropout_rate,
            name="seq_ord_embed" if name else None,
            dtype=self.dtype,
        )

        # Smoothing layer
        if self._enable_res_smoothing:
            self._smooth_seq = [
                keras.layers.Dense(
                    units=output_depth,
                    activation="relu",
                    kernel_initializer="glorot_normal",
                    bias_initializer="zeros",
                    name="res_smooth_dense" if name else None,
                    dtype=self.dtype,
                ),
                keras.layers.Dropout(
                    dropout_rate,
                    name="res_smooth_dropout" if name else None,
                    dtype=self.dtype,
                ),
            ]
        else:
            self._smooth_seq = None

        # Synthesis layers
        self._synth_layers = [
            SynthesisLayer(
                output_depth=output_depth,
                num_heads=num_heads,
                key_dim=key_dim,
                mlp_units=mlp_units,
                dropout_rate=dropout_rate,
                name=f"synth_layer_{i}" if name else None,
                dtype=self.dtype,
            )
            for i in range(self._num_layers)
        ]
        self._last_attn_scores = None

    def call(self, inputs) -> tf.Tensor:
        x, w, condition = inputs
        out = self._seq_ord_embed(x)
        if self._smooth_seq is not None:
            for layer in self._smooth_seq:
                out = layer(out)
        for i in range(self._num_layers):
            out = self._synth_layers[i](out, w, condition)
        self._last_attn_scores = self._synth_layers[-1]._attn_scores
        return out

    @property
    def output_depth(self) -> int:
        return self._synth_layers[0].output_depth

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_heads(self) -> int:
        return self._synth_layers[0].num_heads

    @property
    def key_dim(self) -> int:
        return self._synth_layers[0].key_dim

    @property
    def mlp_units(self) -> int:
        return self._synth_layers[0].mlp_units

    @property
    def dropout_rate(self) -> float:
        return self._seq_ord_embed.dropout_rate

    @property
    def seq_ord_latent_dim(self) -> int:
        return self._seq_ord_embed.latent_dim

    @property
    def seq_ord_max_length(self) -> int:
        return self._seq_ord_embed.max_length

    @property
    def seq_ord_normalization(self) -> float:
        return self._seq_ord_embed.normalization

    @property
    def enable_res_smoothing(self) -> bool:
        return self._enable_res_smoothing
