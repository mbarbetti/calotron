import tensorflow as tf

from calotron.losses import BinaryCrossentropy as BCE
from calotron.losses import JSDivergence as JS_div
from calotron.losses import KLDivergence as KL_div
from calotron.losses import MeanAbsoluteError as MAE
from calotron.losses import MeanSquaredError as MSE
from calotron.losses import WassersteinDistance as Wass_dist
from calotron.losses.BaseLoss import BaseLoss

LOSS_SHORTCUTS = ["bce", "kl_div", "js_div", "mse", "mae", "wass_dist"]
CALOTRON_LOSSES = [BCE(), KL_div(), JS_div(), MSE(), MAE(), Wass_dist()]


def checkLoss(loss) -> BaseLoss:
    if isinstance(loss, str):
        if loss in LOSS_SHORTCUTS:
            for str_loss, calo_loss in zip(LOSS_SHORTCUTS, CALOTRON_LOSSES):
                if loss == str_loss:
                    return calo_loss
        else:
            raise ValueError(
                f"`loss` should be selected in {LOSS_SHORTCUTS}, "
                f"instead '{loss}' passed"
            )
    elif isinstance(loss, BaseLoss):
        return loss
    else:
        raise TypeError(
            f"`loss` should be a calotron's `BaseLoss`, " f"instead {type(loss)} passed"
        )
