import pytest
import numpy as np
import tensorflow as tf


np.random.seed(42)
chunk_size = int(5e4)
y_true = np.random.uniform(0.5, 1.0, size=(chunk_size,))
y_pred = np.random.uniform(0.2, 0.8, size=(chunk_size,))


@pytest.fixture
def metric():
  from calotron.metrics import Accuracy
  metric_ = Accuracy(name="accuracy",
                     dtype=None,
                     threshold=0.5,
                     from_logits=False)
  return metric_


###########################################################################


def test_metric_configuration(metric):
  from calotron.metrics import Accuracy
  assert isinstance(metric, Accuracy)
  assert isinstance(metric.name, str)


def test_metric_use(metric):
  metric.update_state(y_true, y_pred)
  res = metric.result().numpy()
  assert (res > 0.75) and (res < 0.76)


def test_metric_kargs(metric):
  w = np.random.uniform(0.0, 1.0, size=(chunk_size,)) > 0.5
  metric.update_state(y_true, y_pred, sample_weight=w)
  res = metric.result().numpy()
  assert (res > 0.75) and (res < 0.76)
