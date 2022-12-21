import tensorflow as tf
from tensorflow.keras.losses import KLDivergence as TF_KLDivergence
from calotron.losses.BaseLoss import BaseLoss


class JSDivergence(BaseLoss):
  def __init__(self, reduction="auto", name="js_loss"):
    super().__init__(name)
    self._kl_div = TF_KLDivergence(reduction=reduction, name=name)
  
  def __call__(self, y_true, y_pred, **kwargs):
    dtype = self._kl_div(y_true, y_pred).dtype
    y_true = tf.cast(y_true, dtype)
    y_pred = tf.cast(y_pred, dtype)
    loss = 0.5 * self._kl_div(y_true, 0.5 * (y_true + y_pred), **kwargs) + \
           0.5 * self._kl_div(y_pred, 0.5 * (y_true + y_pred), **kwargs)
    return loss

  def discriminator_loss(self, y_true, y_pred, **kwargs):
    return -self(y_true, y_pred, **kwargs)   # divergence maximization

  def transformer_loss(self, y_true, y_pred, **kwargs):
    return self(y_true, y_pred, **kwargs)   # divergence minimization
