import pytest
from calotron.metrics.BaseMetric import BaseMetric
from calotron.metrics import Accuracy
from calotron.metrics import BinaryCrossentropy as BCE
from calotron.metrics import KLDivergence as KL
from calotron.metrics import JSDivergence as JS
from calotron.metrics import MeanSquaredError as MSE
from calotron.metrics import RootMeanSquaredError as RMSE
from calotron.metrics import MeanAbsoluteError as MAE


STR_CASES = ["accuracy", "bce", "kl_div", "js_div", "mse", "rmse", "mae"]
CLS_CASES = [Accuracy(), BCE(), KL(), JS(), MSE(), RMSE(), MAE()]


@pytest.fixture
def checker():
    from calotron.utils import checkMetrics

    chk = checkMetrics
    return chk


###########################################################################


def test_checker_use_None(checker):
    res = checker(None)
    assert res is None


@pytest.mark.parametrize("metrics", [[s] for s in STR_CASES])
def test_checker_use_strings(metrics):
    from calotron.utils import checkMetrics

    res = checkMetrics(metrics)
    assert isinstance(res, list)
    assert len(res) == 1
    for r in res:
        assert isinstance(r, BaseMetric)


@pytest.mark.parametrize("metrics", [[c] for c in CLS_CASES])
def test_checker_use_classes(metrics):
    from calotron.utils import checkMetrics

    res = checkMetrics(metrics)
    assert isinstance(res, list)
    assert len(res) == 1
    for r in res:
        assert isinstance(r, BaseMetric)


def test_checker_use_mixture(checker):
    res = checker(STR_CASES + CLS_CASES)
    assert isinstance(res, list)
    assert len(res) == len(STR_CASES) + len(CLS_CASES)
    for r in res:
        assert isinstance(r, BaseMetric)
