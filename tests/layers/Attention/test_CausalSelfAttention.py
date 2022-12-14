import pytest
import tensorflow as tf


@pytest.fixture
def layer():
  from calotron.layers import CausalSelfAttention
  att = CausalSelfAttention(num_heads=8, key_dim=64)
  return att


###########################################################################


def test_layer_configuration(layer):
  from calotron.layers import CausalSelfAttention
  assert isinstance(layer, CausalSelfAttention)


def test_layer_use(layer):
  input = tf.random.normal(shape=(100, 32, 10))
  output = layer(input)
  assert output.shape == input.shape
  out1 = layer(input[:, :8, :])
  out2 = layer(input)[:, :8, :]
  err = tf.reduce_max(tf.abs(out1 - out2)).numpy()
  assert err < 1e-4
