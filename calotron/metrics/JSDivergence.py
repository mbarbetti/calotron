import tensorflow as tf
from tensorflow.keras.metrics import KLDivergence as TF_KLDivergence
from calotron.metrics.BaseMetric import BaseMetric


class JSDivergence(BaseMetric):
  def __init__(self, name="js_div", dtype=None, **kwargs):
    super().__init__(name, dtype, **kwargs)
    self._kl_div = TF_KLDivergence(name=name, dtype=dtype)

  def __call__(self, y_true, y_pred, **kwargs):
    dtype = self._kl_div(y_true, y_pred).dtype
    y_true = tf.cast(y_true, dtype)
    y_pred = tf.cast(y_pred, dtype)
    metric = 0.5 * self._kl_div(y_true, 0.5 * (y_true + y_pred), **kwargs) + \
             0.5 * self._kl_div(y_pred, 0.5 * (y_true + y_pred), **kwargs)
    return metric