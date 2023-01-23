import numpy as np
import pytest
import tensorflow as tf

np.random.seed(42)
chunk_size = int(1e4)
y_true = None
y_pred = np.random.uniform(0.0, 1.0, size=(chunk_size,))
y_pred_logits = np.random.uniform(-5.0, 5.0, size=(chunk_size,))


@pytest.fixture
def metric():
    from calotron.metrics import BinaryCrossentropy

    metric_ = BinaryCrossentropy(
        name="bce", dtype=None, from_logits=False, label_smoothing=0.0
    )
    return metric_


###########################################################################


def test_metric_configuration(metric):
    from calotron.metrics import BinaryCrossentropy

    assert isinstance(metric, BinaryCrossentropy)
    assert isinstance(metric.name, str)


@pytest.mark.parametrize("from_logits", [False, True])
def test_metric_use_no_weights(from_logits):
    from calotron.metrics import BinaryCrossentropy

    metric = BinaryCrossentropy(
        name="bce", dtype=None, from_logits=from_logits, label_smoothing=0.0
    )
    if from_logits:
        metric.update_state(y_true, y_pred_logits, sample_weight=None)
        res = metric.result().numpy()
        assert (res > 1.2) and (res < 1.5)
    else:
        metric.update_state(y_true, y_pred, sample_weight=None)
        res = metric.result().numpy()
        assert (res > 0.9) and (res < 1.1)


@pytest.mark.parametrize("from_logits", [False, True])
def test_metric_use_with_weights(from_logits):
    from calotron.metrics import BinaryCrossentropy

    metric = BinaryCrossentropy(
        name="bce", dtype=None, from_logits=from_logits, label_smoothing=0.0
    )
    if from_logits:
        metric.update_state(y_true, y_pred_logits, sample_weight=1.0)
        res = metric.result().numpy()
        assert (res > 1.2) and (res < 1.5)
    else:
        metric.update_state(y_true, y_pred, sample_weight=1.0)
        res = metric.result().numpy()
        assert (res > 0.9) and (res < 1.1)
