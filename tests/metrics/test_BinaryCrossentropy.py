import pytest
import numpy as np
import tensorflow as tf


y_true = [[0, 1], [0, 0]]
y_pred = [[0.6, 0.4], [0.4, 0.6]]


@pytest.fixture
def metric():
  from calotron.metrics import BinaryCrossentropy
  metric_ = BinaryCrossentropy(name="bce",
                               dtype=None,
                               from_logits=False,
                               label_smoothing=0.0)
  return metric_


###########################################################################


def test_metric_configuration(metric):
  from calotron.metrics import BinaryCrossentropy
  assert isinstance(metric, BinaryCrossentropy)
  assert isinstance(metric.name, str)


def test_metric_use(metric):
  metric.update_state(y_true, y_pred)
  res = metric.result().numpy()
  assert (res > 0.81) and (res < 0.82)


def test_metric_kargs(metric):
  w = [1, 0]
  metric.update_state(y_true, y_pred, sample_weight=w)
  res = metric.result().numpy()
  assert (res > 0.91) and (res < 0.92)
