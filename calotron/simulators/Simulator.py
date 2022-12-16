import tensorflow as tf
from calotron.models import Transformer


class Simulator(tf.Module):
  def __init__(self, transformer, start_token):
    super().__init__()
    if not isinstance(transformer, Transformer):
      raise TypeError(f"`transformer` should be a calotron's "
                      f"`{type(Transformer).__name__}`, instead "
                      f"{type(transformer)} passed")
    self._transformer = transformer
    if not isinstance(start_token, tf.Tensor):
      raise TypeError(f"`start_token` should be a TensorFlow "
                      f"`Tensor`, instead {type(start_token)} passed")
    self._start_token = start_token

  def __call__(self, source, max_length):
    if not isinstance(source, tf.Tensor):
      raise TypeError(f"`source` should be a TensorFlow "
                      f"`Tensor`, instead {type(source)} passed")
    if max_length < 1:
      raise ValueError("`max_length` should be greater than 0")
    max_length = int(max_length)

    start_token = tf.cast(self._start_token, dtype=source.dtype)
    target = tf.expand_dims(start_token, axis=1)
    for _ in tf.range(max_length):
      predictions = self.transformer([source, target], training=False)
      target = tf.concat([target, predictions[:, -1:, :]], axis=1)

    assert target.shape[1] == max_length + 1
    return target[:, 1:, :]

  @property
  def transformer(self) -> Transformer:
    return self._transformer

  @property
  def start_token(self) -> tf.Tensor:
    return self._start_token
