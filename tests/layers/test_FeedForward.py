import pytest
import tensorflow as tf


@pytest.fixture
def layer():
  from calotron.layers import FeedForward
  ff = FeedForward(output_units=64, hidden_units=128, dropout_rate=0.1)
  return ff


###########################################################################


def test_layer_configuration(layer):
  from calotron.layers import FeedForward
  assert isinstance(layer, FeedForward)
  assert isinstance(layer.output_units, int)
  assert isinstance(layer.hidden_units, int)
  assert isinstance(layer.dropout_rate, float)


def test_layer_use(layer):
  input = tf.keras.Input(shape=[16, 8])
  output = layer(input)
  test_shape = list(input.shape)
  test_shape[-1] = layer.output_units
  assert output.shape == tuple(test_shape)
