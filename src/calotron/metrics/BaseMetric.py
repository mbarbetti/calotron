import tensorflow as tf


class BaseMetric(tf.keras.metrics.Metric):
    def __init__(self, name="metric", dtype=None) -> None:
        super().__init__(name, dtype)
        self._metric_values = self.add_weight(
            name=f"{name}_values", initializer="zeros"
        )

    @staticmethod
    def _prepare_weights(sample_weight=None):
        if sample_weight is not None:
            weights = tf.reduce_mean(sample_weight, axis=1)
        else:
            weights = None
        return weights

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        raise NotImplementedError(
            "Only `BaseMetric` subclasses have the "
            "`update_state()` method implemented."
        )

    def result(self):
        return self._metric_values
