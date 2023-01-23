import tensorflow as tf
from tensorflow.keras.metrics import BinaryAccuracy as TF_BinaryAccuracy
from calotron.metrics.BaseMetric import BaseMetric


class Accuracy(BaseMetric):
    def __init__(self, name="accuracy", dtype=None, threshold=0.5):
        super().__init__(name, dtype)
        self._accuracy = TF_BinaryAccuracy(name=name, dtype=dtype, threshold=threshold)

    def update_state(self, y_true, y_pred, sample_weight=None):
        state = self._accuracy(
            tf.ones_like(y_pred), y_pred, sample_weight=sample_weight
        )
        self._metric_values.assign(state)
