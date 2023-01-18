import pytest
from calotron.losses.BaseLoss import BaseLoss
from calotron.losses import BinaryCrossentropy as BCE
from calotron.losses import KLDivergence as KL
from calotron.losses import JSDivergence as JS
from calotron.losses import MeanSquaredError as MSE
from calotron.losses import MeanAbsoluteError as MAE


STR_CASES = ["bce", "kl", "js", "mse", "mae"]
CLS_CASES = [BCE(), KL(), JS(), MSE(), MAE()]


###########################################################################


@pytest.mark.parametrize("loss", STR_CASES)
def test_checker_use_strings(loss):
  from calotron.utils import checkLoss
  res = checkLoss(loss)
  assert isinstance(res, BaseLoss)


@pytest.mark.parametrize("loss", CLS_CASES)
def test_checker_use_classes(loss):
  from calotron.utils import checkLoss
  res = checkLoss(loss)
  assert isinstance(res, BaseLoss)
