import pytest
import numpy as np
import tensorflow as tf


np.random.seed(42)
chunk_size = int(1e4)
y_true = None
y_pred = np.random.uniform(0.0, 1.0, size=(chunk_size,))


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
  assert (res > 0.9) and (res < 1.1)


@pytest.mark.parametrize("sample_weight", [None, 1.0])
def test_metric_kargs(sample_weight):
  from calotron.metrics import BinaryCrossentropy
  metric = BinaryCrossentropy(name="bce",
                              dtype=None,
                              from_logits=False,
                              label_smoothing=0.0)
  metric.update_state(y_true, y_pred, sample_weight=sample_weight)
  res = metric.result().numpy()
  assert (res > 0.9) and (res < 1.1)
