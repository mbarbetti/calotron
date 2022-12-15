import tensorflow as tf
from calotron.models import Transformer


class BaseSimulator(tf.Module):
  def __init__(self, transformer, start_token):
    super().__init__()
    assert isinstance(transformer, Transformer)
    self._transformer = transformer
    assert isinstance(start_token, tf.Tensor)
    self._start_token = start_token

  def __call__(self, source, max_length):
    assert isinstance(source, tf.Tensor)
    assert max_length > 0

    target = tf.expand_dims(self._start_token, axis=1)
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
