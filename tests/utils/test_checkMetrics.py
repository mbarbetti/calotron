import pytest

from calotron.metrics.BaseMetric import BaseMetric
from calotron.utils.checkMetrics import CALOTRON_METRICS, METRIC_SHORTCUTS


@pytest.fixture
def checker():
    from calotron.utils import checkMetrics

    chk = checkMetrics
    return chk


###########################################################################


def test_checker_use_None(checker):
    res = checker(None)
    assert res is None


@pytest.mark.parametrize("metrics", [[s] for s in METRIC_SHORTCUTS])
def test_checker_use_strings(metrics):
    from calotron.utils import checkMetrics

    res = checkMetrics(metrics)
    assert isinstance(res, list)
    assert len(res) == 1
    for r in res:
        assert isinstance(r, BaseMetric)


@pytest.mark.parametrize("metrics", [[c] for c in CALOTRON_METRICS])
def test_checker_use_classes(metrics):
    from calotron.utils import checkMetrics

    res = checkMetrics(metrics)
    assert isinstance(res, list)
    assert len(res) == 1
    for r in res:
        assert isinstance(r, BaseMetric)


def test_checker_use_mixture(checker):
    res = checker(METRIC_SHORTCUTS + CALOTRON_METRICS)
    assert isinstance(res, list)
    assert len(res) == len(METRIC_SHORTCUTS) + len(CALOTRON_METRICS)
    for r in res:
        assert isinstance(r, BaseMetric)
