import pytest
import numpy as np
import tensorflow as tf


y_true = [[0, 1], [0, 0]]
y_pred = [[0.6, 0.4], [0.4, 0.6]]


@pytest.fixture
def metric():
    from calotron.metrics import JSDivergence

    metric_ = JSDivergence(name="js_div", dtype=None)
    return metric_


###########################################################################


def test_metric_configuration(metric):
    from calotron.metrics import JSDivergence

    assert isinstance(metric, JSDivergence)
    assert isinstance(metric.name, str)


def test_metric_use_no_weights(metric):
    metric.update_state(y_true, y_pred, sample_weight=None)
    res = metric.result().numpy()
    assert (res > 0.33) and (res < 0.34)


def test_metric_use_with_weights(metric):
    w = [1, 0]
    metric.update_state(y_true, y_pred, sample_weight=w)
    res = metric.result().numpy()
    assert (res > 0.39) and (res < 0.40)
