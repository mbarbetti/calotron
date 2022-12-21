import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError as TF_MSE
from calotron.losses.BaseLoss import BaseLoss


class MeanSquaredError(BaseLoss):
  def __init__(self, reduction="auto", name="mse_loss"):
    super().__init__(name)
    self._loss = TF_MSE(reduction=reduction, name=name)
  
  def discriminator_loss(self, y_true, y_pred, **kwargs):
    return -self(y_true, y_pred, **kwargs)   # divergence maximization

  def transformer_loss(self, y_true, y_pred, **kwargs):
    return self(y_true, y_pred, **kwargs)   # divergence minimization
