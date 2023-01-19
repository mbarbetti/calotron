import tensorflow as tf
from calotron.layers.PositionalEmbedding import PositionalEmbedding
from calotron.layers.Attention import CausalSelfAttention
from calotron.layers.Attention import CrossAttention
from calotron.layers.FeedForward import FeedForward


class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, decoder_depth, num_heads, 
               key_dim=None, ff_units=128, dropout_rate=0.1, 
               residual_smoothing=True, name=None, dtype=None):
    super().__init__(name=name, dtype=dtype)
    self._decoder_depth = int(decoder_depth)
    self._num_heads = int(num_heads)
    self._key_dim = int(key_dim) if key_dim else None
    self._ff_units = int(ff_units)
    self._dropout_rate = float(dropout_rate)
    self._residual_smoothing = bool(residual_smoothing)

    self._csa_layer = CausalSelfAttention(
        num_heads=self._num_heads,
        key_dim=self._key_dim if self._key_dim else self._decoder_depth,
        dropout=self._dropout_rate,
        dtype=self.dtype)

    self._ca_layer = CrossAttention(
        num_heads=self._num_heads,
        key_dim=self._key_dim if self._key_dim else self._decoder_depth,
        dropout=self._dropout_rate,
        dtype=self.dtype)

    self._ff_layer = FeedForward(
        output_units=self._decoder_depth, 
        hidden_units=self._ff_units,
        residual_smoothing=self._residual_smoothing,
        dtype=self.dtype)

  def call(self, x, context):
    x = self._csa_layer(x=x)                   # shape: (batch_size, x_elements, x_depth)
    x = self._ca_layer(x=x, context=context)   # shape: (batch_size, x_elements, x_depth)
    x = self._ff_layer(x)                      # shape: (batch_size, x_elements, decoder_depth)
    return x

  @property
  def decoder_depth(self) -> int:
    return self._decoder_depth

  @property
  def num_heads(self) -> int:
    return self._num_heads

  @property
  def key_dim(self):   # TODO: add Union[int, None]
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


class Decoder(tf.keras.layers.Layer):
  def __init__(self, decoder_depth, num_layers, num_heads,
               key_dim=None, pos_dim=None, pos_normalization=512,
               max_length=128, ff_units=128, dropout_rate=0.1,
               pos_sensitive=False, residual_smoothing=True,
               name=None, dtype=None):
    super().__init__(name=name, dtype=dtype)
    self._decoder_depth = int(decoder_depth)
    self._num_layers = int(num_layers)
    self._num_heads = int(num_heads)
    self._key_dim = int(key_dim) if key_dim else None
    self._pos_dim = int(pos_dim) if pos_dim else None
    self._pos_normalization = float(pos_normalization)
    self._max_length = int(max_length)
    self._ff_units = int(ff_units)
    self._dropout_rate = float(dropout_rate)
    self._pos_sensitive = bool(pos_sensitive)
    self._residual_smoothing = bool(residual_smoothing)

    if self._pos_sensitive:
      self._pos_embedding = PositionalEmbedding(
        self._pos_dim if self._pos_dim else self._decoder_depth,
        max_length=self._max_length,
        encoding_normalization=self._pos_normalization,
        dropout_rate=self._dropout_rate,
        dtype=self.dtype)
    else:
      self._pos_embedding = None

    self._dec_layers = [
        DecoderLayer(decoder_depth=self._decoder_depth,
                     num_heads=self._num_heads,
                     key_dim=self._key_dim,
                     ff_units=self._ff_units,
                     dropout_rate=self._dropout_rate,
                     residual_smoothing=self._residual_smoothing,
                     dtype=self.dtype)
        for _ in range(self._num_layers)]

  def call(self, x, context):
    if self._pos_embedding is not None:
      x = self._pos_embedding(x)   # shape: (batch_size, x_elements, pos_dim)
    for i in range(self._num_layers):
      x = self._dec_layers[i](x, context)
    return x   # shape: (batch_size, x_elements, decoder_depth)

  @property
  def decoder_depth(self) -> int:
    return self._decoder_depth

  @property
  def num_layers(self) -> int:
    return self._num_layers
  
  @property
  def num_heads(self) -> int:
    return self._num_heads

  @property
  def key_dim(self):   # TODO: add Union[int, None]
    return self._key_dim

  @property
  def pos_dim(self):   # TODO: add Union[int, None]
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
