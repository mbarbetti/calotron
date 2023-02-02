import pytest

from calotron.losses.BaseLoss import BaseLoss
from calotron.utils.checkLoss import CALOTRON_LOSSES, LOSS_SHORTCUTS

###########################################################################


@pytest.mark.parametrize("loss", LOSS_SHORTCUTS)
def test_checker_use_strings(loss):
    from calotron.utils import checkLoss

    res = checkLoss(loss)
    assert isinstance(res, BaseLoss)


@pytest.mark.parametrize("loss", CALOTRON_LOSSES)
def test_checker_use_classes(loss):
    from calotron.utils import checkLoss

    res = checkLoss(loss)
    assert isinstance(res, BaseLoss)
