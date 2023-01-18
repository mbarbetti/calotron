import pytest
import tensorflow as tf


chunk_size = int(1e5)

source = tf.random.normal(shape=(chunk_size, 2, 5))
target = tf.random.normal(shape=(chunk_size, 4, 10))


@pytest.fixture
def model():
  from calotron.models import Transformer
  trans = Transformer(output_depth=target.shape[2],
                      encoder_depth=8,
                      decoder_depth=8,
                      num_layers=2,
                      num_heads=4,
                      key_dim=None,
                      output_activations=None,
                      ff_units=16,
                      dropout_rate=0.1,
                      residual_smoothing=True)
  return trans


###########################################################################


@pytest.mark.parametrize("key_dim", [None, 64])
@pytest.mark.parametrize("output_activations", [None, "sigmoid"])
def test_model_configuration(key_dim, output_activations):
  from calotron.models import Transformer
  model = Transformer(output_depth=target.shape[2],
                      encoder_depth=8,
                      decoder_depth=8,
                      num_layers=2,
                      num_heads=4,
                      key_dim=key_dim,
                      output_activations=output_activations,
                      ff_units=16,
                      dropout_rate=0.1,
                      residual_smoothing=True)
  assert isinstance(model, Transformer)
  assert isinstance(model.output_depth, int)
  assert isinstance(model.encoder_depth, int)
  assert isinstance(model.decoder_depth, int)
  assert isinstance(model.num_layers, int)
  assert isinstance(model.num_heads, int)
  if model.key_dim is not None:
    assert isinstance(model.key_dim, int)
  if model.output_activations is not None:
    assert isinstance(model.output_activations, list)
  assert isinstance(model.ff_units, int)
  assert isinstance(model.dropout_rate, float)
  assert isinstance(model.residual_smoothing, bool)

  from calotron.layers import Encoder, Decoder
  assert isinstance(model.encoder, Encoder)
  assert isinstance(model.decoder, Decoder)


@pytest.mark.parametrize("key_dim", [None, 64])
@pytest.mark.parametrize("output_activations", [None, "sigmoid"])
@pytest.mark.parametrize("residual_smoothing", [True, False])
def test_model_use(key_dim, output_activations, residual_smoothing):
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
                      output_activations=output_activations,
                      ff_units=16,
                      dropout_rate=0.1,
                      residual_smoothing=residual_smoothing)
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
