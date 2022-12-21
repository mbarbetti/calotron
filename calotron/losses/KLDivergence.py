import tensorflow as tf
from tensorflow.keras.losses import KLDivergence as TF_KLDivergence
from calotron.losses.BaseLoss import BaseLoss


class KLDivergence(BaseLoss):
  def __init__(self, reduction="auto", name="kl_loss"):
    super().__init__(name)
    self._loss = TF_KLDivergence(reduction=reduction, name=name)
  
  def discriminator_loss(self, y_true, y_pred, **kwargs):
    return -self(y_true, y_pred, **kwargs)   # divergence maximization

  def transformer_loss(self, y_true, y_pred, **kwargs):
    return self(y_true, y_pred, **kwargs)   # divergence minimization
