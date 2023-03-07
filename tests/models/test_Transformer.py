import pytest
import tensorflow as tf

from calotron.models.Transformer import START_TOKEN_INITIALIZERS

CHUNK_SIZE = int(1e4)
BATCH_SIZE = 500
ADDITIONAL_DIM = 2


source = tf.random.normal(shape=(CHUNK_SIZE, 16, 3))
target = tf.random.normal(shape=(CHUNK_SIZE, 32, 9))


@pytest.fixture
def model():
    from calotron.models import Transformer

    trans = Transformer(
        output_depth=target.shape[2],
        encoder_depth=8,
        decoder_depth=8,
        num_layers=2,
        num_heads=4,
        key_dims=32,
        fnn_units=128,
        dropout_rates=0.1,
        seq_ord_latent_dims=16,
        seq_ord_max_lengths=[source.shape[1], target.shape[1]],
        seq_ord_normalizations=10_000,
        residual_smoothing=True,
        output_activations="relu",
        start_token_initializer="zeros",
    )
    return trans


###########################################################################


@pytest.mark.parametrize("output_activations", [None, "relu"])
def test_model_configuration(output_activations):
    from calotron.models import Transformer

    model = Transformer(
        output_depth=target.shape[2],
        encoder_depth=8,
        decoder_depth=8,
        num_layers=2,
        num_heads=4,
        key_dims=32,
        fnn_units=128,
        dropout_rates=0.1,
        seq_ord_latent_dims=16,
        seq_ord_max_lengths=[source.shape[1], target.shape[1]],
        seq_ord_normalizations=10_000,
        residual_smoothing=True,
        output_activations=output_activations,
        start_token_initializer="zeros",
    )
    assert isinstance(model, Transformer)
    assert isinstance(model.output_depth, int)
    assert isinstance(model.encoder_depth, int)
    assert isinstance(model.decoder_depth, int)
    assert isinstance(model.num_layers, list)
    assert isinstance(model.num_heads, list)
    assert isinstance(model.key_dims, list)
    assert isinstance(model.fnn_units, list)
    assert isinstance(model.dropout_rates, list)
    assert isinstance(model.seq_ord_latent_dims, list)
    assert isinstance(model.seq_ord_max_lengths, list)
    assert isinstance(model.seq_ord_normalizations, list)
    assert isinstance(model.residual_smoothing, list)
    if model.output_activations is not None:
        assert isinstance(model.output_activations, list)
    assert isinstance(model.start_token_initializer, str)


@pytest.mark.parametrize("residual_smoothing", [True, False])
def test_model_use_residual_smoothing(residual_smoothing):
    latent_dim = 8
    if residual_smoothing:
        encoder_depth = source.shape[2] + latent_dim + ADDITIONAL_DIM
        decoder_depth = target.shape[2] + latent_dim + ADDITIONAL_DIM
    else:
        encoder_depth = source.shape[2] + latent_dim
        decoder_depth = target.shape[2] + latent_dim
    from calotron.models import Transformer

    model = Transformer(
        output_depth=target.shape[2],
        encoder_depth=encoder_depth,
        decoder_depth=decoder_depth,
        num_layers=2,
        num_heads=4,
        key_dims=32,
        fnn_units=128,
        dropout_rates=0.1,
        seq_ord_latent_dims=latent_dim,
        seq_ord_max_lengths=[source.shape[1], target.shape[1]],
        seq_ord_normalizations=10_000,
        residual_smoothing=residual_smoothing,
        output_activations=None,
    )
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


@pytest.mark.parametrize("start_token_initializer", START_TOKEN_INITIALIZERS)
def test_model_use_start_token_initializer(start_token_initializer):
    from calotron.models import Transformer

    model = Transformer(
        output_depth=target.shape[2],
        encoder_depth=8,
        decoder_depth=8,
        num_layers=2,
        num_heads=4,
        key_dims=32,
        fnn_units=128,
        dropout_rates=0.1,
        seq_ord_latent_dims=16,
        seq_ord_max_lengths=[source.shape[1], target.shape[1]],
        seq_ord_normalizations=10_000,
        residual_smoothing=True,
        output_activations="relu",
        start_token_initializer=start_token_initializer,
    )
    output = model((source, target))
    model.summary()
    test_shape = list(target.shape)
    test_shape[-1] = model.output_depth
    assert output.shape == tuple(test_shape)
    start_token = model.get_start_token(target[:BATCH_SIZE])
    test_shape = [BATCH_SIZE, target.shape[2]]
    assert start_token.shape == tuple(test_shape)


def test_model_train(model):
    dataset = (
        tf.data.Dataset.from_tensor_slices(((source, target), target))
        .batch(batch_size=BATCH_SIZE, drop_remainder=True)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    mse = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=adam, loss=mse)
    model.fit(dataset, epochs=3)
