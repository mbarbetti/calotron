import tensorflow as tf
from tensorflow.keras.losses import MeanAbsoluteError as TF_MAE
from calotron.losses.BaseLoss import BaseLoss


class MeanAbsoluteError(BaseLoss):
  def __init__(self, reduction="auto", name="mae_loss"):
    super().__init__(name)
    self._loss = TF_MAE(reduction=reduction, name=name)
  
  def discriminator_loss(self, y_true, y_pred, **kwargs):
    return -self(y_true, y_pred, **kwargs)   # divergence maximization

  def transformer_loss(self, y_true, y_pred, **kwargs):
    return self(y_true, y_pred, **kwargs)   # divergence minimization
