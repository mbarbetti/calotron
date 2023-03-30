import tensorflow as tf

from calotron.metrics.BaseMetric import BaseMetric


class WassersteinDistance(BaseMetric):
    def __init__(self, name="wass_dist", dtype=None) -> None:
        super().__init__(name, dtype)

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        if sample_weight is not None:
            weights = tf.reduce_mean(sample_weight)
            state = tf.reduce_sum(weights * (y_pred - y_true))
            state /= tf.reduce_sum(weights)
        else:
            state = tf.reduce_mean(y_pred - y_true)
        state = tf.cast(state, self.dtype)
        self._metric_values.assign(state)
