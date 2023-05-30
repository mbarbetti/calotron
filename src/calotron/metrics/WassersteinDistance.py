import tensorflow as tf

from calotron.metrics.BaseMetric import BaseMetric


class WassersteinDistance(BaseMetric):
    def __init__(self, name="wass_dist", dtype=None) -> None:
        super().__init__(name, dtype)

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        state = tf.reduce_mean(y_pred - y_true)
        state = tf.cast(state, self.dtype)
        self._metric_values.assign(state)
