import tensorflow as tf

from calotron.layers.Attention import GlobalSelfAttention
from calotron.layers.MultilayerPerceptron import MultilayerPerceptron
from calotron.layers.SeqOrderEmbedding import SeqOrderEmbedding


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        output_depth,
        num_heads,
        key_dim,
        mlp_units=128,
        dropout_rate=0.1,
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

        # Multilayer perceptron units
        assert isinstance(mlp_units, (int, float))
        assert mlp_units >= 1
        self._mlp_units = int(mlp_units)

        # Dropout rate
        assert isinstance(dropout_rate, (int, float))
        assert dropout_rate >= 0.0 and dropout_rate < 1.0
        self._dropout_rate = float(dropout_rate)

        # Multi-head attention
        self._global_attn = GlobalSelfAttention(
            num_heads=self._num_heads,
            key_dim=self._key_dim,
            dropout=self._dropout_rate,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            name=f"{prefix}_global_attn_{suffix}" if name else None,
            dtype=self.dtype,
        )

        # Multilayer perceptron
        self._mlp = MultilayerPerceptron(
            output_units=self._output_depth,
            hidden_units=self._mlp_units,
            dropout_rate=self._dropout_rate,
            name=f"{prefix}_mlp_{suffix}" if name else None,
            dtype=self.dtype,
        )

    def call(self, x, global_attn_mask=None) -> tf.Tensor:
        output = self._global_attn(x, attention_mask=global_attn_mask)
        output = self._mlp(output)
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
    def mlp_units(self) -> int:
        return self._mlp_units

    @property
    def dropout_rate(self) -> float:
        return self._dropout_rate


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self,
        output_depth,
        num_layers,
        num_heads,
        key_dim,
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

        # Multilayer perceptron units
        assert isinstance(mlp_units, (int, float))
        assert mlp_units >= 1
        self._mlp_units = int(mlp_units)

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
        assert isinstance(enable_residual_smoothing, bool)
        self._enable_residual_smoothing = enable_residual_smoothing

        # Sequence order embedding
        self._seq_ord_embedding = SeqOrderEmbedding(
            latent_dim=self._seq_ord_latent_dim,
            max_length=self._seq_ord_max_length,
            normalization=self._seq_ord_normalization,
            dropout_rate=self._dropout_rate,
            name="enc_seq_ord_embedding",
            dtype=self.dtype,
        )

        # Smoothing layer
        if self._enable_residual_smoothing:
            self._smooth_layer = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(
                        units=self._output_depth,
                        activation="linear",
                        kernel_initializer="truncated_normal",
                        bias_initializer="zeros",
                        name="enc_smooth_layer",
                        dtype=self.dtype,
                    ),
                    tf.keras.layers.Dropout(
                        self._dropout_rate, name="enc_dropout", dtype=self.dtype
                    ),
                ]
            )
        else:
            self._smooth_layer = None

        # Encoder layers
        self._encoder_layers = [
            EncoderLayer(
                output_depth=self._output_depth,
                num_heads=self._num_heads,
                key_dim=self._key_dim,
                mlp_units=self._mlp_units,
                dropout_rate=self._dropout_rate,
                name=f"enc_layer_{i}",
                dtype=self.dtype,
            )
            for i in range(self._num_layers)
        ]

    def call(self, x, global_attn_mask=None) -> tf.Tensor:
        output = self._seq_ord_embedding(x)
        if self._smooth_layer is not None:
            output = self._smooth_layer(output)
        for i in range(self._num_layers):
            output = self._encoder_layers[i](output, global_attn_mask)
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
    def mlp_units(self) -> int:
        return self._mlp_units

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
    def enable_residual_smoothing(self) -> bool:
        return self._enable_residual_smoothing
