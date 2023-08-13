import pytest
import tensorflow as tf

CHUNK_SIZE = int(1e4)
BATCH_SIZE = 500

source = tf.random.normal(shape=(CHUNK_SIZE, 8, 5))
weight = tf.random.uniform(shape=(CHUNK_SIZE, source.shape[1]))
labels = tf.random.uniform(shape=(CHUNK_SIZE, 16), minval=0.0, maxval=1.0)


@pytest.fixture
def model():
    from calotron.models.players import ConvDeepSets

    ds = ConvDeepSets(
        latent_dim=labels.shape[-1],
        num_conv_layers=2,
        filters=8,
        kernel_size=2,
        strides=1,
        dropout_rate=0.1,
    )
    return ds


###########################################################################


def test_model_configuration(model):
    from calotron.models.players import ConvDeepSets

    assert isinstance(model, ConvDeepSets)
    assert isinstance(model.latent_dim, int)
    assert isinstance(model.num_conv_layers, int)
    assert isinstance(model.filters, int)
    assert isinstance(model.kernel_size, int)
    assert isinstance(model.strides, int)
    assert isinstance(model.dropout_rate, float)


@pytest.mark.parametrize("padding_mask", [weight, None])
def test_model_use(model, padding_mask):
    output = model(source, padding_mask=padding_mask)
    model.summary()
    test_shape = [source.shape[0]]
    test_shape.append(model.latent_dim)
    assert output.shape == tuple(test_shape)


def test_model_train(model):
    dataset = (
        tf.data.Dataset.from_tensor_slices((source, labels))
        .batch(batch_size=BATCH_SIZE, drop_remainder=True)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    mse = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=adam, loss=mse)
    model.fit(dataset, epochs=1)
