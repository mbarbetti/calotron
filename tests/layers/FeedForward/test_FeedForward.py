import pytest
import tensorflow as tf


@pytest.fixture
def layer():
  from calotron.layers import FeedForward
  ff = FeedForward(output_units=64,
                   hidden_units=128,
                   dropout_rate=0.1,
                   residual_smoothing=True)
  return ff


###########################################################################


def test_layer_configuration(layer):
  from calotron.layers import FeedForward
  assert isinstance(layer, FeedForward)
  assert isinstance(layer.output_units, int)
  assert isinstance(layer.hidden_units, int)
  assert isinstance(layer.dropout_rate, float)
  assert isinstance(layer.residual_smoothing, bool)


@pytest.mark.parametrize("residual_smoothing", [True, False])
def test_layer_use(residual_smoothing):
  if residual_smoothing:
    input_dim, output_dim = (8, 64)
  else:
    input_dim, output_dim = (8, 8)
  from calotron.layers import FeedForward
  layer = FeedForward(output_units=output_dim,
                      hidden_units=128,
                      dropout_rate=0.1,
                      residual_smoothing=residual_smoothing)
  input = tf.keras.Input(shape=[16, input_dim])
  output = layer(input)
  test_shape = list(input.shape)
  test_shape[-1] = layer.output_units
  assert output.shape == tuple(test_shape)
