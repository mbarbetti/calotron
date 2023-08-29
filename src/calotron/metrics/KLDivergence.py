import tensorflow as tf
from tensorflow import keras

from calotron.metrics.BaseMetric import BaseMetric


class KLDivergence(BaseMetric):
    def __init__(self, name="kl_div", dtype=None) -> None:
        super().__init__(name, dtype)
        self._kl_div = keras.metrics.KLDivergence(name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        weights = self._prepare_weights(sample_weight)
        state = self._kl_div(y_true, y_pred, sample_weight=weights)
        self._metric_values.assign(state)
