import tensorflow as tf


class BaseMetric(tf.keras.metrics.Metric):
  def __init__(self, name="metric", dtype=None):
    super().__init__(name, dtype)
    self._metric_values = self.add_weight(name=f"{name}_values",
                                          initializer="zeros")

  def update_state(self, y_true, y_pred, sample_weight=None):
    raise NotImplementedError(f"Only `BaseMetric` subclasses have the "
                              f"`update_state()` method implemented.")

  def result(self):
    return self._metric_values
