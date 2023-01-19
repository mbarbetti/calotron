import pytest
import tensorflow as tf


###########################################################################


@pytest.mark.parametrize("key_dim", [None, 64])
@pytest.mark.parametrize("pos_dim", [None, 9])
def test_layer_configuration(key_dim, pos_dim):
  from calotron.layers import Decoder
  layer = Decoder(decoder_depth=16,
                  num_layers=4, 
                  num_heads=8,
                  key_dim=key_dim,
                  pos_dim=pos_dim,
                  pos_normalization=64,
                  max_length=32,
                  ff_units=128,
                  dropout_rate=0.1,
                  pos_sensitive=False,
                  residual_smoothing=True)
  assert isinstance(layer, Decoder)
  assert isinstance(layer.decoder_depth, int)
  assert isinstance(layer.num_layers, int)
  assert isinstance(layer.num_heads, int)
  if layer.key_dim is not None:
    assert isinstance(layer.key_dim, int)
  if layer.pos_dim is not None:
    assert isinstance(layer.pos_dim, int)
  assert isinstance(layer.pos_normalization, float)
  assert isinstance(layer.max_length, int)
  assert isinstance(layer.ff_units, int)
  assert isinstance(layer.dropout_rate, float)
  assert isinstance(layer.pos_sensitive, bool)
  assert isinstance(layer.residual_smoothing, bool)


@pytest.mark.parametrize("key_dim", [None, 64])
@pytest.mark.parametrize("residual_smoothing", [True, False])
def test_layer_use_no_position(key_dim, residual_smoothing):
  if residual_smoothing:
    input_dim, output_dim = (8, 16)
  else:
    input_dim, output_dim = (9, 9)
  from calotron.layers import Decoder
  layer = Decoder(decoder_depth=output_dim,
                  num_layers=4, 
                  num_heads=8,
                  key_dim=key_dim,
                  pos_dim=None,
                  pos_normalization=64,
                  max_length=32,
                  ff_units=128,
                  dropout_rate=0.1,
                  pos_sensitive=False,
                  residual_smoothing=residual_smoothing)
  source = tf.random.normal(shape=(100, 16, 3))
  target = tf.random.normal(shape=(100, 32, input_dim))
  output = layer(target, source)
  test_shape = list(target.shape)
  test_shape[-1] = layer.decoder_depth
  assert output.shape == tuple(test_shape)


@pytest.mark.parametrize("pos_dim", [None, 9])
@pytest.mark.parametrize("residual_smoothing", [True, False])
def test_layer_use_with_position(pos_dim, residual_smoothing):
  if residual_smoothing:
    input_dim, output_dim = (8, 16)
  else:
    input_dim, output_dim = (9, 9)
  from calotron.layers import Decoder
  layer = Decoder(decoder_depth=output_dim,
                  num_layers=4, 
                  num_heads=8,
                  key_dim=None,
                  pos_dim=pos_dim,
                  pos_normalization=64,
                  max_length=32,
                  ff_units=128,
                  dropout_rate=0.1,
                  pos_sensitive=True,
                  residual_smoothing=residual_smoothing)
  source = tf.random.normal(shape=(100, 16, 3))
  target = tf.random.normal(shape=(100, 32, input_dim))
  output = layer(target, source)
  test_shape = list(target.shape)
  test_shape[-1] = layer.decoder_depth
  assert output.shape == tuple(test_shape)
