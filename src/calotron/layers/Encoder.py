import tensorflow as tf

from calotron.layers.Attention import GlobalSelfAttention
from calotron.layers.FeedForward import FeedForward
from calotron.layers.PositionalEmbedding import PositionalEmbedding


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        encoder_depth,
        num_heads,
        key_dim=None,
        ff_units=256,
        dropout_rate=0.1,
        residual_smoothing=True,
        name=None,
        dtype=None,
    ) -> None:
        super().__init__(name=name, dtype=dtype)
        assert encoder_depth >= 1
        self._encoder_depth = int(encoder_depth)
        assert num_heads >= 1
        self._num_heads = int(num_heads)
        if key_dim:
            assert key_dim >= 1
        self._key_dim = int(key_dim) if key_dim else None
        assert ff_units >= 1
        self._ff_units = int(ff_units)
        assert dropout_rate >= 0.0
        self._dropout_rate = float(dropout_rate)
        assert isinstance(residual_smoothing, bool)
        self._residual_smoothing = residual_smoothing

        self._gsa_layer = GlobalSelfAttention(
            num_heads=self._num_heads,
            key_dim=self._key_dim if self._key_dim else self._encoder_depth,
            dropout=self._dropout_rate,
            dtype=self.dtype,
        )

        self._ff_layer = FeedForward(
            output_units=self._encoder_depth,
            hidden_units=self._ff_units,
            residual_smoothing=self._residual_smoothing,
            dtype=self.dtype,
        )

    def call(self, x) -> tf.Tensor:
        x = self._gsa_layer(x)  # (batch_size, x_elements, x_depth)
        x = self._ff_layer(x)  # (batch_size, x_elements, encoder_depth)
        return x

    @property
    def encoder_depth(self) -> int:
        return self._encoder_depth

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def key_dim(self):  # TODO: add Union[int, None]
        return self._key_dim

    @property
    def ff_units(self) -> int:
        return self._ff_units

    @property
    def dropout_rate(self) -> float:
        return self._dropout_rate

    @property
    def residual_smoothing(self) -> bool:
        return self._residual_smoothing


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self,
        encoder_depth,
        num_layers,
        num_heads,
        key_dim=None,
        pos_dim=None,
        pos_normalization=128,
        max_length=32,
        ff_units=256,
        dropout_rate=0.1,
        pos_sensitive=False,
        residual_smoothing=True,
        name=None,
        dtype=None,
    ) -> None:
        super().__init__(name=name, dtype=dtype)
        assert encoder_depth >= 1
        self._encoder_depth = int(encoder_depth)
        assert num_layers >= 1
        self._num_layers = int(num_layers)
        assert num_heads >= 1
        self._num_heads = int(num_heads)
        if key_dim:
            assert key_dim >= 1
        self._key_dim = int(key_dim) if key_dim else None
        if pos_dim:
            assert pos_dim >= 1
        self._pos_dim = int(pos_dim) if pos_dim else None
        assert pos_normalization > 0.0
        self._pos_normalization = float(pos_normalization)
        assert max_length >= 1
        self._max_length = int(max_length)
        assert ff_units >= 1
        self._ff_units = int(ff_units)
        assert dropout_rate >= 0.0
        self._dropout_rate = float(dropout_rate)
        assert isinstance(pos_sensitive, bool)
        self._pos_sensitive = pos_sensitive
        assert isinstance(residual_smoothing, bool)
        self._residual_smoothing = residual_smoothing

        if self._pos_sensitive:
            self._pos_embedding = PositionalEmbedding(
                self._pos_dim if self._pos_dim else self._encoder_depth,
                max_length=self._max_length,
                encoding_normalization=self._pos_normalization,
                dropout_rate=self._dropout_rate,
                dtype=self.dtype,
            )
        else:
            self._pos_embedding = None

        self._enc_layers = [
            EncoderLayer(
                encoder_depth=self._encoder_depth,
                num_heads=self._num_heads,
                key_dim=self._key_dim,
                ff_units=self._ff_units,
                dropout_rate=self._dropout_rate,
                residual_smoothing=self._residual_smoothing,
                dtype=self.dtype,
            )
            for _ in range(self._num_layers)
        ]

    def call(self, x) -> tf.Tensor:
        if self._pos_embedding is not None:
            x = self._pos_embedding(x)  # (batch_size, x_elements, pos_dim)
        for i in range(self._num_layers):
            x = self._enc_layers[i](x)
        return x  # (batch_size, x_elements, encoder_depth)

    @property
    def encoder_depth(self) -> int:
        return self._encoder_depth

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def key_dim(self):  # TODO: add Union[int, None]
        return self._key_dim

    @property
    def pos_dim(self):  # TODO: add Union[int, None]
        return self._pos_dim

    @property
    def pos_normalization(self) -> float:
        return self._pos_normalization

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def ff_units(self) -> int:
        return self._ff_units

    @property
    def dropout_rate(self) -> float:
        return self._dropout_rate

    @property
    def pos_sensitive(self) -> bool:
        return self._pos_sensitive

    @property
    def residual_smoothing(self) -> bool:
        return self._residual_smoothing
