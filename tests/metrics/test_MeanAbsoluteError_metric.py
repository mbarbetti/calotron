import pytest
import numpy as np
import tensorflow as tf


y_true = [[0, 1], [0, 0]]
y_pred = [[1, 1], [0, 0]]


@pytest.fixture
def metric():
  from calotron.metrics import MeanAbsoluteError
  metric_ = MeanAbsoluteError(name="mae", dtype=None)
  return metric_


###########################################################################


def test_metric_configuration(metric):
  from calotron.metrics import MeanAbsoluteError
  assert isinstance(metric, MeanAbsoluteError)
  assert isinstance(metric.name, str)


def test_metric_use_no_weights(metric):
  metric.update_state(y_true, y_pred, sample_weight=None)
  res = metric.result().numpy()
  assert res == 0.25


def test_metric_use_with_weights(metric):
  w = [1, 0]
  metric.update_state(y_true, y_pred, sample_weight=w)
  res = metric.result().numpy()
  assert res == 0.5
