import pytest
import tensorflow as tf


chunk_size = int(1e4)

source = tf.random.normal(shape=(chunk_size, 16, 3))
target = tf.random.normal(shape=(chunk_size, 32, 9))


@pytest.fixture
def model():
  from calotron.models import Transformer
  trans = Transformer(output_depth=target.shape[2],
                      encoder_depth=8,
                      decoder_depth=8,
                      num_layers=2,
                      num_heads=4,
                      key_dim=None,
                      encoder_pos_dim=None,
                      decoder_pos_dim=None,
                      encoder_pos_normalization=64,
                      decoder_pos_normalization=64,
                      encoder_max_length=source.shape[1], 
                      decoder_max_length=target.shape[1],
                      ff_units=16,
                      dropout_rate=0.1,
                      pos_sensitive=False,
                      residual_smoothing=True,
                      output_activations="relu")
  return trans


###########################################################################


@pytest.mark.parametrize("key_dim", [None, 64])
@pytest.mark.parametrize("encoder_pos_dim", [None, 3])
@pytest.mark.parametrize("decoder_pos_dim", [None, 9])
@pytest.mark.parametrize("output_activations", [None, "relu"])
def test_model_configuration(key_dim, encoder_pos_dim,
                             decoder_pos_dim, output_activations):
  from calotron.models import Transformer
  model = Transformer(output_depth=target.shape[2],
                      encoder_depth=8,
                      decoder_depth=8,
                      num_layers=2,
                      num_heads=4,
                      key_dim=key_dim,
                      encoder_pos_dim=encoder_pos_dim,
                      decoder_pos_dim=decoder_pos_dim,
                      encoder_pos_normalization=64,
                      decoder_pos_normalization=64,
                      encoder_max_length=source.shape[1], 
                      decoder_max_length=target.shape[1],
                      ff_units=16,
                      dropout_rate=0.1,
                      pos_sensitive=False,
                      residual_smoothing=True,
                      output_activations=output_activations)
  assert isinstance(model, Transformer)
  assert isinstance(model.output_depth, int)
  assert isinstance(model.encoder_depth, int)
  assert isinstance(model.decoder_depth, int)
  assert isinstance(model.num_layers, int)
  assert isinstance(model.num_heads, int)
  if model.key_dim is not None:
    assert isinstance(model.key_dim, int)
  if model.pos_dim is not None:
    assert isinstance(model.pos_dim, tuple)
  assert isinstance(model.pos_normalization, tuple)
  assert isinstance(model.max_length, tuple)
  assert isinstance(model.ff_units, int)
  assert isinstance(model.dropout_rate, float)
  assert isinstance(model.pos_sensitive, bool)
  assert isinstance(model.residual_smoothing, bool)
  if model.output_activations is not None:
    assert isinstance(model.output_activations, list)

  from calotron.layers import Encoder, Decoder
  assert isinstance(model.encoder, Encoder)
  assert isinstance(model.decoder, Decoder)


@pytest.mark.parametrize("key_dim", [None, 64])
@pytest.mark.parametrize("residual_smoothing", [True, False])
def test_model_use_no_position(key_dim, residual_smoothing):
  if residual_smoothing:
    encoder_depth, decoder_depth = (8, 16)
  else:
    encoder_depth, decoder_depth = (source.shape[2], target.shape[2])
  from calotron.models import Transformer
  model = Transformer(output_depth=target.shape[2],
                      encoder_depth=encoder_depth,
                      decoder_depth=decoder_depth,
                      num_layers=2,
                      num_heads=4,
                      key_dim=key_dim,
                      encoder_pos_dim=None,
                      decoder_pos_dim=None,
                      encoder_pos_normalization=64,
                      decoder_pos_normalization=64,
                      encoder_max_length=source.shape[1], 
                      decoder_max_length=target.shape[1],
                      ff_units=16,
                      dropout_rate=0.1,
                      pos_sensitive=False,
                      residual_smoothing=residual_smoothing,
                      output_activations=None)
  output = model((source, target))
  model.summary()
  test_shape = list(target.shape)
  test_shape[-1] = model.output_depth
  assert output.shape == tuple(test_shape)


@pytest.mark.parametrize("encoder_pos_dim", [None, 3])
@pytest.mark.parametrize("decoder_pos_dim", [None, 9])
@pytest.mark.parametrize("residual_smoothing", [True, False])
def test_model_use_with_position(encoder_pos_dim,
                                 decoder_pos_dim,
                                 residual_smoothing):
  if residual_smoothing:
    encoder_depth, decoder_depth = (8, 16)
  else:
    encoder_depth, decoder_depth = (source.shape[2], target.shape[2])
  from calotron.models import Transformer
  model = Transformer(output_depth=target.shape[2],
                      encoder_depth=encoder_depth,
                      decoder_depth=decoder_depth,
                      num_layers=2,
                      num_heads=4,
                      key_dim=None,
                      encoder_pos_dim=encoder_pos_dim,
                      decoder_pos_dim=decoder_pos_dim,
                      encoder_pos_normalization=64,
                      decoder_pos_normalization=64,
                      encoder_max_length=source.shape[1], 
                      decoder_max_length=target.shape[1],
                      ff_units=16,
                      dropout_rate=0.1,
                      pos_sensitive=True,
                      residual_smoothing=residual_smoothing,
                      output_activations=None)
  output = model((source, target))
  model.summary()
  test_shape = list(target.shape)
  test_shape[-1] = model.output_depth
  assert output.shape == tuple(test_shape)


def test_model_use_multi_activations(model):
  output = model((source, target))
  model.summary()
  test_shape = list(target.shape)
  test_shape[-1] = model.output_depth
  assert output.shape == tuple(test_shape)


def test_model_train(model):
  dataset = (
      tf.data.Dataset.from_tensor_slices(((source, target), target))
          .batch(batch_size=512, drop_remainder=True)
          .cache()
          .prefetch(tf.data.AUTOTUNE)
    )
  adam = tf.keras.optimizers.Adam(learning_rate=0.001)
  mse = tf.keras.losses.MeanSquaredError()
  model.compile(optimizer=adam, loss=mse)
  model.fit(dataset, epochs=3)
