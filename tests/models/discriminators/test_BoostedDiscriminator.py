import pytest
import tensorflow as tf

CHUNK_SIZE = int(1e4)

source = tf.random.normal(shape=(CHUNK_SIZE, 8, 5))
target = tf.random.normal(shape=(CHUNK_SIZE, 4, 3))
labels = tf.random.uniform(shape=(CHUNK_SIZE,), minval=0.0, maxval=1.0)
labels = tf.cast(labels > 0.5, target.dtype)


@pytest.fixture
def model():
    from calotron.models.discriminators import BoostedDiscriminator

    disc = BoostedDiscriminator(
        output_units=1,
        output_activation="sigmoid",
        latent_dim=8,
        deepsets_num_layers=2,
        deepsets_hidden_units=32,
        dropout_rate=0.1,
    )
    return disc


###########################################################################


def test_model_configuration(model):
    from calotron.models.discriminators import BoostedDiscriminator

    assert isinstance(model, BoostedDiscriminator)
    assert isinstance(model.output_units, int)
    assert isinstance(model.latent_dim, int)
    assert isinstance(model.deepsets_num_layers, int)
    assert isinstance(model.deepsets_hidden_units, int)
    assert isinstance(model.dropout_rate, float)


@pytest.mark.parametrize("activation", ["sigmoid", None])
def test_model_use(activation):
    from calotron.models.discriminators import BoostedDiscriminator

    model = BoostedDiscriminator(
        output_units=1,
        output_activation=activation,
        latent_dim=8,
        deepsets_num_layers=2,
        deepsets_hidden_units=32,
        dropout_rate=0.1,
    )
    output = model((source, target))
    model.summary()
    test_shape = [target.shape[0]]
    test_shape.append(model.output_units)
    assert output.shape == tuple(test_shape)


def test_model_train(model):
    dataset = (
        tf.data.Dataset.from_tensor_slices(((source, target), labels))
        .batch(batch_size=512, drop_remainder=True)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    model.compile(optimizer=adam, loss=bce)
    model.fit(dataset, epochs=2)