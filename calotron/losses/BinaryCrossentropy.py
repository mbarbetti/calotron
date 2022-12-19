import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy as TF_BCE
from calotron.losses.BaseLoss import BaseLoss


class BinaryCrossentropy(BaseLoss):
  def __init__(self,
               from_logits=False,
               label_smoothing=0.0,
               axis=-1,
               reduction="auto",
               name="bce_loss"):
    super().__init__(name)
    self._loss = TF_BCE(from_logits=from_logits,
                        label_smoothing=label_smoothing,
                        axis=axis,
                        reduction=reduction,
                        name=name)
  
  def discriminator_loss(self, y_true, y_pred, **kwargs):
    return -self._loss(y_true, y_pred, **kwargs)   # loss maximization

  def transformer_loss(self, y_true, y_pred, **kwargs):
    return self._loss(y_true, y_pred, **kwargs)   # loss minimization
