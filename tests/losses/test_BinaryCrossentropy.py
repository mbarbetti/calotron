import pytest
import numpy as np
import tensorflow as tf


y_true = [[0, 1], [0, 0]]
y_pred = [[-18.6, 0.51], [2.94, -12.8]]


@pytest.fixture
def loss():
  from calotron.losses import BinaryCrossentropy
  loss_ = BinaryCrossentropy(from_logits=False,
                             label_smoothing=0.0,
                             axis=-1,
                             reduction="auto",
                             name="bce_loss")
  return loss_


###########################################################################


def test_loss_configuration(loss):
  from calotron.losses import BinaryCrossentropy
  assert isinstance(loss, BinaryCrossentropy)
  assert isinstance(loss.name, str)


@pytest.mark.parametrize("from_logits", [False, True])
def test_loss_use(from_logits):
  from calotron.losses import BinaryCrossentropy
  loss = BinaryCrossentropy(from_logits=from_logits,
                            label_smoothing=0.0,
                            axis=-1,
                            reduction="auto",
                            name="bce_loss")
  out1 = loss.discriminator_loss(y_true, y_pred).numpy()
  out2 = loss.transformer_loss(y_true, y_pred).numpy()
  assert out1 == -out2


@pytest.mark.parametrize("reduction",
                         [tf.keras.losses.Reduction.SUM,
                          tf.keras.losses.Reduction.NONE])
def test_loss_reduction(reduction):
  from calotron.losses import BinaryCrossentropy
  loss = BinaryCrossentropy(from_logits=True,
                            label_smoothing=0.0,
                            axis=-1,
                            reduction=reduction,
                            name="bce_loss")
  out1 = loss.discriminator_loss(y_true, y_pred).numpy()
  out2 = loss.transformer_loss(y_true, y_pred).numpy()
  assert np.all(np.equal(out1, -out2))


def test_loss_kargs(loss):
  w = [0.8, 0.2]
  out1 = loss.discriminator_loss(y_true, y_pred, sample_weight=w).numpy()
  out2 = loss.transformer_loss(y_true, y_pred, sample_weight=w).numpy()
  assert out1 == - out2
