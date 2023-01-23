import numpy as np
import pytest
import tensorflow as tf

y_true = [[0, 1], [0, 0]]
y_pred = [[1, 1], [0, 0]]


@pytest.fixture
def metric():
    from calotron.metrics import RootMeanSquaredError

    metric_ = RootMeanSquaredError(name="rmse", dtype=None)
    return metric_


###########################################################################


def test_metric_configuration(metric):
    from calotron.metrics import RootMeanSquaredError

    assert isinstance(metric, RootMeanSquaredError)
    assert isinstance(metric.name, str)


def test_metric_use_no_weights(metric):
    metric.update_state(y_true, y_pred, sample_weight=None)
    res = metric.result().numpy()
    assert res == 0.5


def test_metric_use_with_weights(metric):
    w = [1, 0]
    metric.update_state(y_true, y_pred, sample_weight=w)
    res = metric.result().numpy()
    assert (res > 0.70) and (res < 0.71)
