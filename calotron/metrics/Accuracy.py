import tensorflow as tf
from tensorflow.keras.metrics import BinaryAccuracy as TF_BinaryAccuracy
from calotron.metrics.BaseMetric import BaseMetric


class Accuracy(BaseMetric):
  def __init__(self,
               name="accuracy",
               dtype=None,
               threshold=0.5,
               from_logits=False):
    super().__init__(name, dtype)
    self._accuracy = TF_BinaryAccuracy(name=name,
                                       dtype=dtype,
                                       threshold=threshold)
    self._from_logits = bool(from_logits)

  def update_state(self, y_true, y_pred, sample_weight=None):
    if self._from_logits:
      y_true = tf.sigmoid(y_true)
      y_pred = tf.sigmoid(y_pred)
    y_tot = tf.concat([y_true, y_pred], axis=0)
    if sample_weight is not None:
      w_tot = tf.concat([sample_weight, sample_weight], axis=0)
    else:
      w_tot = None
    state = self._accuracy(tf.ones_like(y_tot), y_tot, sample_weight=w_tot)
    self._metric_values.assign(state)
