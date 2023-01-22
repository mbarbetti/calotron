import pytest
import numpy as np
import tensorflow as tf


y_true = [[0, 1], [0, 0]]
y_pred = [[0.6, 0.4], [0.4, 0.6]]


@pytest.fixture
def metric():
  from calotron.metrics import KLDivergence
  metric_ = KLDivergence(name="kl_div", dtype=None)
  return metric_


###########################################################################


def test_metric_configuration(metric):
  from calotron.metrics import KLDivergence
  assert isinstance(metric, KLDivergence)
  assert isinstance(metric.name, str)


def test_metric_use_no_weights(metric):
  metric.update_state(y_true, y_pred, sample_weight=None)
  res = metric.result().numpy()
  assert (res > 0.45) and (res < 0.46)


def test_metric_use_with_weights(metric):
  w = [1, 0]
  metric.update_state(y_true, y_pred, sample_weight=w)
  res = metric.result().numpy()
  assert (res > 0.91) and (res < 0.92)
