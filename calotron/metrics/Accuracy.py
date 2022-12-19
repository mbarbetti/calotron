import tensorflow as tf
from tensorflow.keras.metrics import Accuracy as TF_Accuracy
from calotron.metrics.BaseMetric import BaseMetric


class Accuracy(BaseMetric):
  def __init__(self, name="accuracy", dtype=None, **kwargs):
    super().__init__(name, dtype)
    self._metric = TF_Accuracy(name=name, dtype=dtype, **kwargs)
