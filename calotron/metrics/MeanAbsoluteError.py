import tensorflow as tf
from tensorflow.keras.metrics import MeanAbsoluteError as TF_MAE
from calotron.metrics.BaseMetric import BaseMetric


class MeanAbsoluteError(BaseMetric):
  def __init__(self, name="mae", dtype=None, **kwargs):
    super().__init__(name, dtype, **kwargs)
    self._metric = TF_MAE(name=name, dtype=dtype)
