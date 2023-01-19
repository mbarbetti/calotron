import tensorflow as tf
from calotron.utils import checkActivations


class MultiActivations(tf.keras.layers.Layer):
  def __init__(self, activations, output_depth, name=None, dtype=None):
    super().__init__(name=name, dtype=dtype)
    self._output_depth = int(output_depth)
    self._output_activations = checkActivations(activations,
                                                output_depth,
                                                dtype)

  def call(self, x):
    if x.shape[2] != self._output_depth:
      raise ValueError(f"`output_depth` passed {self._output_depth} "
                       f"doesn't match with the input tensor shape "
                       f"({x.shape}).")
    if self._output_activations is not None:
      concat = list()
      for i, activation in enumerate(self._output_activations):
        concat.append(activation(x[:,:,i])[:,:,None])
      x = tf.concat(concat, axis=2)
    return x   # shape (batch_size, x_elements, x_depth)

  @property
  def output_activations(self):   # TODO: add Union[list, None]
    return self._output_activations

  @property
  def output_depth(self) -> int:
    return self._output_depth
