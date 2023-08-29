import tensorflow as tf
from tensorflow import keras

from calotron.metrics.BaseMetric import BaseMetric


class Accuracy(BaseMetric):
    def __init__(self, name="accuracy", dtype=None, threshold=0.5) -> None:
        super().__init__(name, dtype)
        self._accuracy = keras.metrics.BinaryAccuracy(
            name=name, dtype=dtype, threshold=threshold
        )

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        weights = self._prepare_weights(sample_weight)
        state = self._accuracy(tf.ones_like(y_pred), y_pred, sample_weight=weights)
        self._metric_values.assign(state)
