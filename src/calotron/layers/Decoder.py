import tensorflow as tf

from calotron.layers.Attention import CausalSelfAttention, CrossAttention
from calotron.layers.FeedForward import FeedForward
from calotron.layers.SeqOrderEmbedding import SeqOrderEmbedding


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        output_depth,
        num_heads,
        key_dim,
        fnn_units=128,
        dropout_rate=0.1,
        residual_smoothing=True,
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

        # Feed-forward net units
        assert isinstance(fnn_units, (int, float))
        assert fnn_units >= 1
        self._fnn_units = int(fnn_units)

        # Dropout rate
        assert isinstance(dropout_rate, (int, float))
        assert dropout_rate >= 0.0 and dropout_rate < 1.0
        self._dropout_rate = float(dropout_rate)

        # Residual smoothing
        assert isinstance(residual_smoothing, bool)
        self._residual_smoothing = residual_smoothing

        # Masked multi-head attention
        self._causal_attn = CausalSelfAttention(
            num_heads=self._num_heads,
            key_dim=self._key_dim,
            kernel_initializer="glorot_uniform",
            dropout=self._dropout_rate,
            name=f"{prefix}_causal_attn_{suffix}" if name else None,
            dtype=self.dtype,
        )

        # Multi-head attention
        self._cross_attn = CrossAttention(
            num_heads=self._num_heads,
            key_dim=self._key_dim,
            kernel_initializer="glorot_uniform",
            dropout=self._dropout_rate,
            name=f"{prefix}_cross_attn_{suffix}" if name else None,
            dtype=self.dtype,
        )

        # Feed-forward net
        self._ff_net = FeedForward(
            output_units=self._output_depth,
            hidden_units=self._fnn_units,
            dropout_rate=self._dropout_rate,
            residual_smoothing=self._residual_smoothing,
            name=f"{prefix}_fnn_{suffix}" if name else None,
            dtype=self.dtype,
        )

    def call(self, x, condition) -> tf.Tensor:
        x = self._causal_attn(x)
        output = self._cross_attn(x, condition)
        output = self._ff_net(output)
        return output

    @property
    def output_depth(self) -> int:
        return self._output_depth

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def key_dim(self):  # TODO: add Union[int, None]
        return self._key_dim

    @property
    def fnn_units(self) -> int:
        return self._fnn_units

    @property
    def dropout_rate(self) -> float:
        return self._dropout_rate

    @property
    def residual_smoothing(self) -> bool:
        return self._residual_smoothing


class Decoder(tf.keras.layers.Layer):
    def __init__(
        self,
        output_depth,
        num_layers,
        num_heads,
        key_dim,
        fnn_units=128,
        dropout_rate=0.1,
        seq_ord_latent_dim=16,
        seq_ord_max_length=512,
        seq_ord_normalization=10_000,
        residual_smoothing=True,
        name=None,
        dtype=None,
    ) -> None:
        super().__init__(name=name, dtype=dtype)

        # Output depth
        assert isinstance(output_depth, (int, float))
        assert output_depth >= 1
        self._output_depth = int(output_depth)

        # Number of layers
        assert isinstance(num_layers, (int, float))
        assert num_layers >= 1
        self._num_layers = int(num_layers)

        # Number of heads
        assert isinstance(num_heads, (int, float))
        assert num_heads >= 1
        self._num_heads = int(num_heads)

        # Key dimension
        assert isinstance(key_dim, (int, float))
        assert key_dim >= 1
        self._key_dim = int(key_dim)

        # Feed-forward net units
        assert isinstance(fnn_units, (int, float))
        assert fnn_units >= 1
        self._fnn_units = int(fnn_units)

        # Dropout rate
        assert isinstance(dropout_rate, (int, float))
        assert dropout_rate >= 0.0 and dropout_rate < 1.0
        self._dropout_rate = float(dropout_rate)

        # Sequence order latent space dimension
        assert isinstance(seq_ord_latent_dim, (int, float))
        assert seq_ord_latent_dim >= 1
        self._seq_ord_latent_dim = int(seq_ord_latent_dim)

        # Sequence max length
        assert isinstance(seq_ord_max_length, (int, float))
        assert seq_ord_max_length >= 1
        self._seq_ord_max_length = int(seq_ord_max_length)

        # Sequence order encoding normalization
        assert isinstance(seq_ord_normalization, (int, float))
        assert seq_ord_normalization > 0.0
        self._seq_ord_normalization = float(seq_ord_normalization)

        # Residual smoothing
        assert isinstance(residual_smoothing, bool)
        self._residual_smoothing = residual_smoothing

        # Sequence order embedding
        self._seq_ord_embedding = SeqOrderEmbedding(
            latent_dim=self._seq_ord_latent_dim,
            max_length=self._seq_ord_max_length,
            normalization=self._seq_ord_normalization,
            name="dec_seq_ord_embedding",
            dtype=self.dtype,
        )

        # Dropout layer
        self._dropout = tf.keras.layers.Dropout(
            self._dropout_rate, name="dec_dropout", dtype=self.dtype
        )

        # Decoder layers
        self._decoder_layers = [
            DecoderLayer(
                output_depth=self._output_depth,
                num_heads=self._num_heads,
                key_dim=self._key_dim,
                fnn_units=self._fnn_units,
                dropout_rate=self._dropout_rate,
                residual_smoothing=self._residual_smoothing,
                name=f"dec_layer_{i}",
                dtype=self.dtype,
            )
            for i in range(self._num_layers)
        ]

    def call(self, x, condition) -> tf.Tensor:
        seq_order = self._seq_ord_embedding(x)
        output = tf.concat([x, seq_order], axis=2)
        output = self._dropout(output)
        for i in range(self._num_layers):
            output = self._decoder_layers[i](output, condition)
        return output

    @property
    def output_depth(self) -> int:
        return self._output_depth

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def key_dim(self) -> int:
        return self._key_dim

    @property
    def fnn_units(self) -> int:
        return self._fnn_units

    @property
    def dropout_rate(self) -> float:
        return self._dropout_rate

    @property
    def seq_ord_latent_dim(self) -> int:
        return self._seq_ord_latent_dim

    @property
    def seq_ord_max_length(self) -> int:
        return self._seq_ord_max_length

    @property
    def seq_ord_normalization(self) -> float:
        return self._seq_ord_normalization

    @property
    def residual_smoothing(self) -> bool:
        return self._residual_smoothing
