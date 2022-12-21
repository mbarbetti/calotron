import pytest
import numpy as np
import tensorflow as tf


y_true = [[0, 1], [0, 0]]
y_pred = [[0.6, 0.4], [0.4, 0.6]]


@pytest.fixture
def loss():
  from calotron.losses import JSDivergence
  loss_ = JSDivergence(reduction="auto", name="js_loss")
  return loss_


###########################################################################


def test_loss_configuration(loss):
  from calotron.losses import JSDivergence
  assert isinstance(loss, JSDivergence)
  assert isinstance(loss.name, str)


def test_loss_configuration(loss):
  out1 = loss.discriminator_loss(y_true, y_pred).numpy()
  out2 = loss.transformer_loss(y_true, y_pred).numpy()
  assert out1 == -out2


@pytest.mark.parametrize("reduction",
                         [tf.keras.losses.Reduction.SUM,
                          tf.keras.losses.Reduction.NONE])
def test_loss_reduction(reduction):
  from calotron.losses import JSDivergence
  loss = JSDivergence(reduction=reduction, name="js_loss")
  out1 = loss.discriminator_loss(y_true, y_pred).numpy()
  out2 = loss.transformer_loss(y_true, y_pred).numpy()
  assert np.all(np.equal(out1, -out2))


def test_loss_kargs(loss):
  w = [0.8, 0.2]
  out1 = loss.discriminator_loss(y_true, y_pred, sample_weight=w).numpy()
  out2 = loss.transformer_loss(y_true, y_pred, sample_weight=w).numpy()
  assert out1 == - out2
