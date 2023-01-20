import numpy as np
import tensorflow as tf


class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, output_depth, max_length=128, 
               encoding_normalization=512, dropout_rate=0.1,
               name=None, dtype=None):
    super().__init__(name=name, dtype=dtype)
    self._output_depth = int(output_depth)
    self._max_length = int(max_length)
    self._encoding_normalization = float(encoding_normalization)
    self._dropout_rate = float(dropout_rate)

    self._embedding = tf.keras.layers.Dense(self._output_depth,
                                            activation="linear",
                                            dtype=self.dtype)

    self._pos_encoding = self._positional_encoding(
                             length=self._max_length,
                             depth=self._output_depth,
                             normalization=self._encoding_normalization,
                             dtype=self.dtype)

    self._dropout = tf.keras.layers.Dropout(self._dropout_rate,
                                            dtype=self.dtype)

  def call(self, x):
    length = tf.shape(x)[1]
    x = self._embedding(x)
    x *= tf.math.sqrt(tf.cast(self._output_depth, self.dtype))   # scale factor
    x = x + self._pos_encoding[None, :length, :]
    x = self._dropout(x)
    return x

  @staticmethod
  def _positional_encoding(length, depth,
                           normalization=512,
                           dtype=tf.float32):
    pos_encoding = np.zeros(shape=(length, depth))   # buffer to fill
    for k in range(length):
      for i in range(int(depth/2)):
        denominator = np.power(normalization, 2*i/depth)
        pos_encoding[k, 2*i] = np.sin(k/denominator)
        pos_encoding[k, 2*i+1] = np.cos(k/denominator)
    return tf.cast(pos_encoding, dtype=dtype)

  @property
  def output_depth(self) -> int:
    return self._output_depth

  @property
  def max_length(self) -> int:
    return self._max_length

  @property
  def encoding_normalization(self) -> float:
    return self._encoding_normalization

  @property
  def dropout_rate(self) -> float:
    return self._dropout_rate
