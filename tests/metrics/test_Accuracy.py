import pytest
import numpy as np
import tensorflow as tf


y_true = [[1], [2], [3], [4]]
y_pred = [[0], [2], [3], [4]]


@pytest.fixture
def metric():
  from calotron.metrics import Accuracy
  metric_ = Accuracy(name="accuracy", dtype=None)
  return metric_


###########################################################################


def test_metric_configuration(metric):
  from calotron.metrics import Accuracy
  assert isinstance(metric, Accuracy)
  assert isinstance(metric.name, str)


def test_metric_use(metric):
  metric.update_state(y_true, y_pred)
  res = metric.result().numpy()
  assert res == 0.75


def test_metric_kargs(metric):
  w = [1, 1, 0, 0]
  metric.update_state(y_true, y_pred, sample_weight=w)
  res = metric.result().numpy()
  assert res == 0.5
