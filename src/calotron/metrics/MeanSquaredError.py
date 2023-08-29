import tensorflow as tf
from tensorflow import keras

from calotron.metrics.BaseMetric import BaseMetric


class MeanSquaredError(BaseMetric):
    def __init__(self, name="mse", dtype=None, **kwargs) -> None:
        super().__init__(name, dtype, **kwargs)
        self._mse = keras.metrics.MeanSquaredError(name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        weights = self._prepare_weights(sample_weight)
        state = self._mse(y_true, y_pred, sample_weight=weights)
        self._metric_values.assign(state)
