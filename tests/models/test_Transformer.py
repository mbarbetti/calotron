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
                      ff_units=16,
                      dropout_rate=0.1)
  return trans


###########################################################################


def test_model_configuration(model):
  from calotron.models import Transformer
  assert isinstance(model, Transformer)
  assert isinstance(model.output_depth, int)
  assert isinstance(model.encoder_depth, int)
  assert isinstance(model.decoder_depth, int)
  assert isinstance(model.num_layers, int)
  assert isinstance(model.num_heads, int)
  if model.key_dim is not None:
    assert isinstance(model.key_dim, int)
  assert isinstance(model.ff_units, int)
  assert isinstance(model.dropout_rate, float)

  from calotron.layers import Encoder, Decoder
  assert isinstance(model.encoder, Encoder)
  assert isinstance(model.decoder, Decoder)


@pytest.mark.parametrize("key_dim", [None, 8])
def test_model_use(key_dim):
  from calotron.models import Transformer
  model = Transformer(output_depth=target.shape[2],
                      encoder_depth=8,
                      decoder_depth=8,
                      num_layers=2,
                      num_heads=4,
                      key_dim=key_dim,
                      ff_units=16,
                      dropout_rate=0.1)
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
