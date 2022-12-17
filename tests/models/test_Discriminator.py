import pytest
import tensorflow as tf


chunk_size = int(5e4)

input1 = tf.random.normal(shape=(chunk_size, 4, 10), mean=1.0)
input2 = tf.random.normal(shape=(chunk_size, 4, 10), mean=2.0)
inputs = tf.concat([input1, input2], axis=0)

label1 = tf.zeros(shape=(chunk_size,))
label2 = tf.ones(shape=(chunk_size,))
labels = tf.concat([label1, label2], axis=0)


@pytest.fixture
def model():
  from calotron.models import Discriminator
  disc = Discriminator(latent_dim=8,
                       output_units=1,
                       output_activation="sigmoid",
                       hidden_layers=2,
                       hidden_units=32,
                       dropout_rate=0.1)
  return disc


###########################################################################


def test_model_configuration(model):
  from calotron.models import Discriminator
  assert isinstance(model, Discriminator)
  assert isinstance(model.latent_dim, int)
  assert isinstance(model.output_units, int)
  assert isinstance(model.hidden_layers, int)
  assert isinstance(model.hidden_units, int)
  assert isinstance(model.dropout_rate, float)

  from calotron.layers import DeepSets
  assert isinstance(model.deepsets, DeepSets)


@pytest.mark.parametrize("activation", ["sigmoid", None])
def test_model_use(activation):
  from calotron.models import Discriminator
  model = Discriminator(latent_dim=8,
                        output_units=1,
                        output_activation=activation,
                        hidden_layers=2,
                        hidden_units=32,
                        dropout_rate=0.1)
  output = model(inputs)
  model.summary()
  test_shape = [inputs.shape[0]]
  test_shape.append(model.output_units)
  assert output.shape == tuple(test_shape)


def test_model_train(model):
  dataset = (
      tf.data.Dataset.from_tensor_slices((inputs, labels))
          .batch(batch_size=512, drop_remainder=True)
          .cache()
          .prefetch(tf.data.AUTOTUNE)
    )
  adam = tf.keras.optimizers.Adam(learning_rate=0.001)
  bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
  model.compile(optimizer=adam, loss=bce)
  model.fit(dataset, epochs=3)
