import tensorflow as tf

from calotron.metrics.BaseMetric import BaseMetric


class WassersteinDistance(BaseMetric):
    def __init__(self, name="wass_dist", dtype=None) -> None:
        super().__init__(name, dtype)

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        weights = self._prepare_weights(sample_weight)
        if weights is not None:
            state = tf.reduce_sum(weights * (y_true - y_pred)) / tf.reduce_sum(weights)
        else:
            state = tf.reduce_mean(y_true - y_pred)
        state = tf.cast(state, self.dtype)
        self._metric_values.assign(state)
