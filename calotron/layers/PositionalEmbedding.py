import numpy as np
import tensorflow as tf


class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, output_depth, max_length=128,
               encoding_normalization=512,
               name=None, dtype=None):
    super().__init__(name=name, dtype=dtype)
    self._output_depth = int(output_depth)
    self._max_length = int(max_length)
    self._encoding_normalization = float(encoding_normalization)

    self._embedding = tf.keras.layers.Embedding(input_dim=self._max_length,
                                                output_dim=self._output_depth,
                                                mask_zero=True)

    self._pos_encoding = self._positional_encoding(
                                length=self._max_length,
                                depth=self._output_depth,
                                normalization=self._encoding_normalization,
                                dtype=self.dtype)

  def compute_mask(self, *args, **kwargs):
    return self._embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    length = tf.shape(x)[1]
    x = self._embedding(x)
    x *= tf.math.sqrt(tf.cast(self._output_depth, self.dtype))   # scale factor
    x = x + self._pos_encoding[None, :length, :]
    return x

  @staticmethod
  def _positional_encoding(length, depth,
                           normalization=512,
                           dtype=tf.float32):
    depth = int(depth/2)
    depths = np.arange(depth)[None, :]/depth    # shape: (1, depth)
    angle_rates = 1 / (normalization**depths)   # shape: (1, depth)

    positions = np.arange(length)[:, None]      # shape: (seq, 1)
    angle_rads = positions * angle_rates        # shape: (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1) 

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
