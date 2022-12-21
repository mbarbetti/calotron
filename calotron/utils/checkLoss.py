import tensorflow as tf
from calotron.losses.BaseLoss import BaseLoss
from calotron.losses import BinaryCrossentropy as BCE
from calotron.losses import KLDivergence as KL
from calotron.losses import JSDivergence as JS
from calotron.losses import MeanSquaredError as MSE
from calotron.losses import MeanAbsoluteError as MAE


LOSS_SHORTCUTS = ["bce", "kl", "js", "mse", "mae"]
CALOTRON_LOSSES = [BCE(), KL(), JS(), MSE(), MAE()]


def checkLoss(loss) -> BaseLoss:
  if isinstance(loss, str):
    if loss in LOSS_SHORTCUTS:
      for str_loss, calo_loss in zip(LOSS_SHORTCUTS, CALOTRON_LOSSES):
        if loss == str_loss:
          return calo_loss
    else:
      raise ValueError(f"`loss` should be selected in {LOSS_SHORTCUTS}, "
                       f"instead '{loss}' passed")
  elif isinstance(loss, BaseLoss):
    return loss
  else:
    raise TypeError(f"`loss` should be a calotron's `BaseLoss`, "
                    f"instead {type(loss)} passed")
