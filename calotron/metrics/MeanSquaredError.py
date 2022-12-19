import tensorflow as tf
from tensorflow.keras.metrics import MeanSquaredError as TF_MSE
from calotron.metrics.BaseMetric import BaseMetric


class MeanSquaredError(BaseMetric):
  def __init__(self, name="mse", dtype=None, **kwargs):
    super().__init__(name, dtype, **kwargs)
    self._metric = TF_MSE(name=name, dtype=dtype)
