import pytest
import tensorflow as tf


@pytest.fixture
def layer():
  from calotron.layers import PositionalEmbedding
  pe = PositionalEmbedding(output_depth=8,
                           max_length=128,
                           encoding_normalization=512)
  return pe


###########################################################################


def test_layer_configuration(layer):
  from calotron.layers import PositionalEmbedding
  assert isinstance(layer, PositionalEmbedding)
  assert isinstance(layer.output_depth, int)
  assert isinstance(layer.max_length, int)
  assert isinstance(layer.encoding_normalization, float)


def test_layer_use(layer):
  input = tf.random.uniform(shape=(32, 10))   # TODO: understand how to use it
  output = layer(input)
  test_shape = list(input.shape)
  test_shape.append(layer.output_depth)
  assert output.shape == tuple(test_shape)
