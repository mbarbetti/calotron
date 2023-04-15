import pytest
import tensorflow as tf

from calotron.models.transformers import Transformer

CHUNK_SIZE = int(1e4)

source = tf.random.normal(shape=(CHUNK_SIZE, 8, 5))
target = tf.random.normal(shape=(CHUNK_SIZE, 4, 3))
labels = tf.random.uniform(shape=(CHUNK_SIZE, 8), minval=0.0, maxval=1.0)
labels = tf.cast(labels > 0.5, source.dtype)

transf = Transformer(
    output_depth=target.shape[2],
    encoder_depth=8,
    decoder_depth=8,
    num_layers=2,
    num_heads=4,
    key_dims=32,
    mlp_units=16,
    dropout_rates=0.1,
    seq_ord_latent_dims=16,
    seq_ord_max_lengths=[source.shape[1], target.shape[1]],
    seq_ord_normalizations=10_000,
    enable_residual_smoothing=True,
    output_activations="relu",
    start_token_initializer="ones",
)


@pytest.fixture
def model():
    from calotron.models.auxiliaries import AuxClassifier

    aux = AuxClassifier(
        transformer=transf,
        output_depth=1,
        output_activation="sigmoid",
        dropout_rate=0.1,
    )
    return aux


###########################################################################


def test_model_configuration(model):
    from calotron.models.auxiliaries import AuxClassifier

    assert isinstance(model, AuxClassifier)
    assert isinstance(model.output_depth, int)
    assert isinstance(model.dropout_rate, float)


@pytest.mark.parametrize("activation", ["sigmoid", None])
def test_model_use(activation):
    from calotron.models.auxiliaries import AuxClassifier

    model = AuxClassifier(
        transformer=transf,
        output_depth=1,
        output_activation=activation,
        dropout_rate=0.1,
    )
    output = model(source)
    model.summary()
    test_shape = [source.shape[0], source.shape[1]]
    test_shape.append(model.output_depth)
    assert output.shape == tuple(test_shape)


def test_model_train(model):
    dataset = (
        tf.data.Dataset.from_tensor_slices((source, labels))
        .batch(batch_size=512, drop_remainder=True)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    model.compile(optimizer=adam, loss=bce)
    model.fit(dataset, epochs=2)
