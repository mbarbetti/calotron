import tensorflow as tf
from calotron.layers import Encoder, Decoder


class Transformer(tf.keras.Model):
  def __init__(self, 
               output_depth, 
               encoder_depth, 
               decoder_depth, 
               num_layers, 
               num_heads, 
               key_dim=None,
               ff_units=128, 
               dropout_rate=0.1):
    super().__init__()
    self._output_depth = int(output_depth)
    self._encoder_depth = int(encoder_depth)
    self._decoder_depth = int(decoder_depth)
    self._num_layers = int(num_layers)
    self._num_heads = int(num_heads)
    self._key_dim = int(key_dim) if key_dim else None
    self._ff_units = int(ff_units)
    self._dropout_rate = float(dropout_rate)

    self._encoder = Encoder(encoder_depth=self._encoder_depth,
                            num_layers=self._num_layers,
                            num_heads=self._num_heads, 
                            key_dim=self._key_dim,
                            ff_units=self._ff_units,
                            dropout_rate=self._dropout_rate)

    self._decoder = Decoder(decoder_depth=self._decoder_depth,
                            num_layers=self._num_layers,
                            num_heads=self._num_heads,
                            key_dim=self._key_dim,
                            ff_units=self._ff_units,
                            dropout_rate=self._dropout_rate)

    self._final_layer = tf.keras.layers.Dense(self._output_depth)

  def call(self, inputs):
    source, target = inputs
    context = self._encoder(x=source)                   # shape: (batch_size, source_elements, encoder_depth)
    output = self._decoder(x=target, context=context)   # shape: (batch_size, target_elements, decoder_depth)
    output = self._final_layer(output)                  # shape: (batch_size, target_elements, output_depth)
    return output

  @property
  def output_depth(self) -> int:
    return self._output_depth

  @property
  def encoder_depth(self) -> int:
    return self._encoder_depth

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
  def ff_units(self) -> int:
    return self._ff_units

  @property
  def dropout_rate(self) -> float:
    return self._dropout_rate

  @property
  def encoder(self) -> Encoder:
    return self._encoder

  @property
  def decoder(self) -> Decoder:
    return self._decoder
