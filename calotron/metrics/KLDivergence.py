import tensorflow as tf
from tensorflow.keras.metrics import KLDivergence as TF_KLDivergence
from calotron.metrics.BaseMetric import BaseMetric


class KLDivergence(BaseMetric):
  def __init__(self, name="kl_div", dtype=None, **kwargs):
    super().__init__(name, dtype, **kwargs)
    self._metric = TF_KLDivergence(name=name, dtype=dtype)
