import pytest
import tensorflow as tf


@pytest.fixture
def layer():
  from calotron.layers import Encoder
  enc_layer = Encoder(encoder_depth=16, num_layers=4, 
                      num_heads=8, key_dim=64, ff_units=128)
  return enc_layer


###########################################################################


def test_layer_configuration(layer):
  from calotron.layers import Encoder
  assert isinstance(layer, Encoder)
  assert isinstance(layer.encoder_depth, int)
  assert isinstance(layer.num_layers, int)
  assert isinstance(layer.num_heads, int)
  if layer.key_dim is not None:
    assert isinstance(layer.key_dim, int)
  assert isinstance(layer.ff_units, int)
  assert isinstance(layer.dropout_rate, float)


@pytest.mark.parametrize("key_dim", [None, 64])
def test_layer_use(key_dim):
  from calotron.layers import Encoder
  layer = Encoder(encoder_depth=16,
                  num_layers=4, 
                  num_heads=8,
                  key_dim=key_dim,
                  ff_units=128)
  input = tf.random.normal(shape=(100, 32, 10))
  output = layer(input)
  test_shape = list(input.shape)
  test_shape[-1] = layer.encoder_depth
  assert output.shape == tuple(test_shape)
