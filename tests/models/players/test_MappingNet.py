import pytest
import tensorflow as tf

CHUNK_SIZE = int(1e4)
BATCH_SIZE = 500
ADDITIONAL_DIM = 2

source = tf.random.normal(shape=(CHUNK_SIZE, 8))
target = tf.random.normal(shape=(CHUNK_SIZE, 4))


@pytest.fixture
def model():
    from calotron.models.players import MappingNet

    map = MappingNet(
        output_dim=target.shape[-1],
        latent_dim=64,
        num_layers=4,
        hidden_units=128,
        dropout_rate=0.1,
    )
    return map


###########################################################################


def test_model_configuration(model):
    from calotron.models.players import MappingNet

    assert isinstance(model, MappingNet)
    assert isinstance(model.output_dim, int)
    assert isinstance(model.latent_dim, int)
    assert isinstance(model.num_layers, int)
    assert isinstance(model.hidden_units, int)
    assert isinstance(model.dropout_rate, float)


def test_model_use(model):
    output = model(source)
    model.summary()
    test_shape = list(source.shape)
    test_shape[-1] = model.output_dim
    assert output.shape == tuple(test_shape)


def test_model_train(model):
    dataset = (
        tf.data.Dataset.from_tensor_slices((source, target))
        .batch(batch_size=BATCH_SIZE, drop_remainder=True)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    mse = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=adam, loss=mse)
    model.fit(dataset, epochs=1)
