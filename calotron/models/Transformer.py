import tensorflow as tf
from calotron.layers import Encoder, Decoder, MultiActivations


class Transformer(tf.keras.Model):
  def __init__(self, 
               output_depth, 
               encoder_depth, 
               decoder_depth, 
               num_layers, 
               num_heads, 
               key_dim=None,
               encoder_pos_dim=None,
               decoder_pos_dim=None,
               encoder_pos_normalization=512,
               decoder_pos_normalization=512,
               encoder_max_length=128,
               decoder_max_length=128,
               ff_units=128, 
               dropout_rate=0.1,
               pos_sensitive = False,
               residual_smoothing=True,
               output_activations=None,
               name=None,
               dtype=None):
    super().__init__(name=name, dtype=dtype)
    self._output_depth = int(output_depth)
    self._encoder_depth = int(encoder_depth)
    self._decoder_depth = int(decoder_depth)
    self._num_layers = int(num_layers)
    self._num_heads = int(num_heads)
    self._key_dim = int(key_dim) if key_dim else None
    self._pos_dim = (int(encoder_pos_dim) if encoder_pos_dim else None,
                     int(decoder_pos_dim) if decoder_pos_dim else None)
    self._pos_normalization = (float(encoder_pos_normalization),
                               float(decoder_pos_normalization))
    self._max_length = (int(encoder_max_length),
                        int(decoder_max_length))
    self._ff_units = int(ff_units)
    self._dropout_rate = float(dropout_rate)
    self._pos_sensitive = bool(pos_sensitive)
    self._residual_smoothing = bool(residual_smoothing)

    self._encoder = Encoder(encoder_depth=self._encoder_depth,
                            num_layers=self._num_layers,
                            num_heads=self._num_heads, 
                            key_dim=self._key_dim,
                            pos_dim=self._pos_dim[0],
                            pos_normalization=self._pos_normalization[0],
                            max_length=self._max_length[0],
                            ff_units=self._ff_units,
                            dropout_rate=self._dropout_rate,
                            pos_sensitive=self._pos_sensitive,
                            residual_smoothing=self._residual_smoothing,
                            dtype=self.dtype)

    self._decoder = Decoder(decoder_depth=self._decoder_depth,
                            num_layers=self._num_layers,
                            num_heads=self._num_heads,
                            key_dim=self._key_dim,
                            pos_dim=self._pos_dim[1],
                            pos_normalization=self._pos_normalization[1],
                            max_length=self._max_length[1],
                            ff_units=self._ff_units,
                            dropout_rate=self._dropout_rate,
                            pos_sensitive=self._pos_sensitive,
                            residual_smoothing=self._residual_smoothing,
                            dtype=self.dtype)

    self._final_layer = tf.keras.layers.Dense(self._output_depth,
                                              name="output_layer",
                                              dtype=self.dtype)

    if output_activations is not None:
      # TODO: find a way to remove the whole list of activations from the summary
      self._multi_act_layer = MultiActivations(output_activations, 
                                               self._output_depth,
                                               name="ma_layer",
                                               dtype=self.dtype)
      self._output_activations = self._multi_act_layer.output_activations
    else:
      self._output_activations = None

  def call(self, inputs):
    source, target = inputs
    context = self._encoder(x=source)                   # shape: (batch_size, source_elements, encoder_depth)
    output = self._decoder(x=target, context=context)   # shape: (batch_size, target_elements, decoder_depth)
    output = self._final_layer(output)                  # shape: (batch_size, target_elements, output_depth)
    if self._output_activations is not None:
      output = self._multi_act_layer(output)            # shape: (batch_size, target_elements, output_depth)
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
  def pos_dim(self):   # TODO: add Union[tuple, None]
    return self._pos_dim

  @property
  def pos_normalization(self) -> tuple:
    return self._pos_normalization

  @property
  def max_length(self) -> tuple:
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

  @property
  def output_activations(self):   # TODO: add Union[list, None]
    return self._output_activations

  @property
  def encoder(self) -> Encoder:
    return self._encoder

  @property
  def decoder(self) -> Decoder:
    return self._decoder
