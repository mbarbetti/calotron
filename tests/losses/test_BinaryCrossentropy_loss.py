import pytest
import numpy as np
import tensorflow as tf


np.random.seed(42)
chunk_size = int(5e4)
y_true = np.random.uniform(0.5, 1.0, size=(chunk_size,))
y_pred = np.random.uniform(0.2, 0.8, size=(chunk_size,))


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
  assert out1 > out2


@pytest.mark.parametrize("reduction",
                         [tf.keras.losses.Reduction.SUM,
                          tf.keras.losses.Reduction.NONE])
def test_loss_reduction(reduction):
  from calotron.losses import BinaryCrossentropy
  loss = BinaryCrossentropy(from_logits=False,
                            label_smoothing=0.0,
                            axis=-1,
                            reduction=reduction,
                            name="bce_loss")
  out1 = loss.discriminator_loss(y_true, y_pred).numpy()
  out2 = loss.transformer_loss(y_true, y_pred).numpy()


def test_loss_kargs(loss):
  w = np.random.uniform(0.0, 1.0, size=(chunk_size,)) > 0.5
  out1 = loss.discriminator_loss(y_true, y_pred, sample_weight=w[None,:]).numpy()
  out2 = loss.transformer_loss(y_true, y_pred, sample_weight=w[None,:]).numpy()
  assert out1 > out2
