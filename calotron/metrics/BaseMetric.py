from tensorflow.keras.metrics import Metric as TF_Metric


class BaseMetric(TF_Metric):
  def __init__(self, name="metric", dtype=None, **kwargs):
    super().__init__(name, dtype, **kwargs)
    if not isinstance(name, str):
      raise TypeError(f"`name` should be a string "
                      f"instead {type(name)} passed")
    self._name = name
    self._metric = None
    self._metric_values = self.add_weight(name="mv", initializer="zeros")

  def __call__(self, y_true, y_pred, **kwargs):
    self._result = self._metric(y_true, y_pred, **kwargs)
    return self._result

  def update_state(self, y_true, y_pred, **kwargs):
    self._metric_values.assign_add(self(y_true, y_pred, **kwargs))

  def result(self):
    return self._metric_values

  @property
  def name(self) -> str:
    return self._name
