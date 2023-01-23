import pytest
from tensorflow.keras.optimizers import Optimizer, SGD, RMSprop, Adam


STR_CASES = ["sgd", "rmsprop", "adam"]
CLS_CASES = [SGD(), RMSprop(), Adam()]


###########################################################################


@pytest.mark.parametrize("optimizer", STR_CASES)
def test_checker_use_strings(optimizer):
    from calotron.utils import checkOptimizer

    res = checkOptimizer(optimizer)
    assert isinstance(res, Optimizer)


@pytest.mark.parametrize("optimizer", CLS_CASES)
def test_checker_use_classes(optimizer):
    from calotron.utils import checkOptimizer

    res = checkOptimizer(optimizer)
    assert isinstance(res, Optimizer)
