import tensorflow as tf
from tensorflow.keras.metrics import RootMeanSquaredError as TF_RMSE
from calotron.metrics.BaseMetric import BaseMetric


class RootMeanSquaredError(BaseMetric):
  def __init__(self, name="rmse", dtype=None, **kwargs):
    super().__init__(name, dtype, **kwargs)
    self._metric = TF_RMSE(name=name, dtype=dtype)
