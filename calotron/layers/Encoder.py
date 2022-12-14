import tensorflow as tf
from calotron.layers import GlobalSelfAttention, FeedForward


class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, encoder_depth, num_heads, 
               key_dim=None, ff_units=128, dropout_rate=0.1):
    super().__init__()
    assert encoder_depth > 0
    self._encoder_depth = int(encoder_depth)
    assert num_heads > 0
    self._num_heads = int(num_heads)
    if key_dim is not None:
      assert key_dim > 0
    self._key_dim = int(key_dim) if key_dim else None
    assert ff_units > 0
    self._ff_units = int(ff_units)
    assert (dropout_rate >= 0.0) and (dropout_rate <= 1.0)
    self._dropout_rate = float(dropout_rate)

    self._gsa_layer = GlobalSelfAttention(
        num_heads=self._num_heads,
        key_dim=self._key_dim if self._key_dim else self._encoder_depth,
        dropout=self._dropout_rate)

    self._ff_layer = FeedForward(
        output_units=self._encoder_depth, 
        hidden_units=self._ff_units)

  def call(self, x):
    x = self._gsa_layer(x)   # shape: (batch_size, x_elements, x_depth)
    x = self._ff_layer(x)    # shape: (batch_size, x_elements, encoder_depth)
    return x

  @property
  def encoder_depth(self) -> int:
    return self._encoder_depth

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


class Encoder(tf.keras.layers.Layer):
  def __init__(self, encoder_depth, num_layers, num_heads, 
               key_dim=None, ff_units=128, dropout_rate=0.1):
    super().__init__()
    assert encoder_depth > 0
    self._encoder_depth = int(encoder_depth)
    assert num_layers > 0
    self._num_layers = int(num_layers)
    assert num_heads > 0
    self._num_heads = int(num_heads)
    if key_dim is not None:
      assert key_dim > 0
    self._key_dim = int(key_dim) if key_dim else None
    assert ff_units > 0
    self._ff_units = int(ff_units)
    assert (dropout_rate >= 0.0) and (dropout_rate <= 1.0)
    self._dropout_rate = float(dropout_rate)

    self._enc_layers = [
        EncoderLayer(encoder_depth=self._encoder_depth,
                     num_heads=self._num_heads,
                     key_dim=self._key_dim,
                     ff_units=self._ff_units,
                     dropout_rate=self._dropout_rate)
        for _ in range(num_layers)]

  def call(self, x):
    for i in range(self._num_layers):
      x = self._enc_layers[i](x)
    return x   # shape: (batch_size, x_elements, encoder_depth)

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
  def key_dim(self):   # TODO: add Union[int, None]
    return self._key_dim

  @property
  def ff_units(self) -> int:
    return self._ff_units

  @property
  def dropout_rate(self) -> float:
    return self._dropout_rate
