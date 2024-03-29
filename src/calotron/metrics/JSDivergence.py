import tensorflow as tf
from tensorflow import keras

from calotron.metrics.BaseMetric import BaseMetric


class JSDivergence(BaseMetric):
    def __init__(self, name="js_div", dtype=None) -> None:
        super().__init__(name, dtype)
        self._kl_div = keras.metrics.KLDivergence(name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        dtype = self._kl_div(y_true, y_pred).dtype
        y_true = tf.cast(y_true, dtype)
        y_pred = tf.cast(y_pred, dtype)

        weights = self._prepare_weights(sample_weight)
        state = 0.5 * self._kl_div(
            y_true, 0.5 * (y_true + y_pred), sample_weight=weights
        ) + 0.5 * self._kl_div(y_pred, 0.5 * (y_true + y_pred), sample_weight=weights)
        self._metric_values.assign(state)
