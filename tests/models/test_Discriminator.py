import pytest
import tensorflow as tf

CHUNK_SIZE = int(5e4)

input1 = tf.random.normal(shape=(CHUNK_SIZE, 4, 10), mean=1.0)
input2 = tf.random.normal(shape=(CHUNK_SIZE, 4, 10), mean=2.0)
inputs = tf.concat([input1, input2], axis=0)

label1 = tf.zeros(shape=(CHUNK_SIZE,))
label2 = tf.ones(shape=(CHUNK_SIZE,))
labels = tf.concat([label1, label2], axis=0)


@pytest.fixture
def model():
    from calotron.models import Discriminator

    disc = Discriminator(
        output_units=1,
        output_activation="sigmoid",
        latent_dim=8,
        hidden_layers=2,
        hidden_units=32,
        dropout_rate=0.1,
    )
    return disc


###########################################################################


def test_model_configuration(model):
    from calotron.models import Discriminator

    assert isinstance(model, Discriminator)
    assert isinstance(model.output_units, int)
    assert isinstance(model.latent_dim, int)
    assert isinstance(model.hidden_layers, int)
    assert isinstance(model.hidden_units, int)
    assert isinstance(model.dropout_rate, float)


@pytest.mark.parametrize("activation", ["sigmoid", None])
def test_model_use(activation):
    from calotron.models import Discriminator

    model = Discriminator(
        output_units=1,
        output_activation=activation,
        latent_dim=8,
        hidden_layers=2,
        hidden_units=32,
        dropout_rate=0.1,
    )
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
