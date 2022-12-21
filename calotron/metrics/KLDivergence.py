import tensorflow as tf
from tensorflow.keras.metrics import KLDivergence as TF_KLDivergence
from calotron.metrics.BaseMetric import BaseMetric


class KLDivergence(BaseMetric):
  def __init__(self, name="kl_div", dtype=None, from_logits=False):
    super().__init__(name, dtype)
    self._kl_div = TF_KLDivergence(name=name, dtype=dtype)
    self._from_logits = bool(from_logits)

  def update_state(self, y_true, y_pred, sample_weight=None):
    if self._from_logits:
      y_true = tf.sigmoid(y_true)
      y_pred = tf.sigmoid(y_pred)
    state = self._kl_div(y_true, y_pred, sample_weight=sample_weight)
    self._metric_values.assign(state)
