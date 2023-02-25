import pytest
import tensorflow as tf

from calotron.models.Transformer import START_TOKEN_INITIALIZERS

CHUNK_SIZE = int(1e4)
BATCH_SIZE = 500


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
        key_dims=None,
        pos_dims=None,
        pos_norms=64,
        max_lengths=[source.shape[1], target.shape[1]],
        ff_units=16,
        dropout_rates=0.1,
        pos_sensitive=False,
        residual_smoothing=True,
        output_activations="relu",
        start_token_initializer="zeros",
    )
    return trans


###########################################################################


@pytest.mark.parametrize("key_dims", [None, 64, [None, 64], [64, 64]])
@pytest.mark.parametrize("pos_dims", [None, 6, [None, 6], [6, 6]])
@pytest.mark.parametrize("output_activations", [None, "relu"])
def test_model_configuration(key_dims, pos_dims, output_activations):
    from calotron.models import Transformer

    model = Transformer(
        output_depth=target.shape[2],
        encoder_depth=8,
        decoder_depth=8,
        num_layers=2,
        num_heads=4,
        key_dims=key_dims,
        pos_dims=pos_dims,
        pos_norms=64,
        max_lengths=[source.shape[1], target.shape[1]],
        ff_units=16,
        dropout_rates=0.1,
        pos_sensitive=False,
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
    assert isinstance(model.pos_dims, list)
    assert isinstance(model.pos_norms, list)
    assert isinstance(model.max_lengths, list)
    assert isinstance(model.ff_units, list)
    assert isinstance(model.dropout_rates, list)
    assert isinstance(model.pos_sensitive, list)
    assert isinstance(model.residual_smoothing, list)
    if model.output_activations is not None:
        assert isinstance(model.output_activations, list)
    assert isinstance(model.start_token_initializer, str)


@pytest.mark.parametrize("key_dims", [None, 64, [None, 64], [64, 64]])
@pytest.mark.parametrize("residual_smoothing", [True, False])
def test_model_use_no_position(key_dims, residual_smoothing):
    if residual_smoothing:
        encoder_depth, decoder_depth = (8, 16)
    else:
        encoder_depth, decoder_depth = (source.shape[2], target.shape[2])
    from calotron.models import Transformer

    model = Transformer(
        output_depth=target.shape[2],
        encoder_depth=encoder_depth,
        decoder_depth=decoder_depth,
        num_layers=2,
        num_heads=4,
        key_dims=key_dims,
        pos_dims=None,
        pos_norms=64,
        max_lengths=[source.shape[1], target.shape[1]],
        ff_units=16,
        dropout_rates=0.1,
        pos_sensitive=False,
        residual_smoothing=residual_smoothing,
        output_activations=None,
    )
    output = model((source, target))
    model.summary()
    test_shape = list(target.shape)
    test_shape[-1] = model.output_depth
    assert output.shape == tuple(test_shape)


@pytest.mark.parametrize("pos_dims", [None, [source.shape[2], target.shape[2]]])
@pytest.mark.parametrize("residual_smoothing", [True, False])
def test_model_use_with_position(pos_dims, residual_smoothing):
    if residual_smoothing:
        encoder_depth, decoder_depth = (8, 16)
    else:
        encoder_depth, decoder_depth = (source.shape[2], target.shape[2])
    from calotron.models import Transformer

    model = Transformer(
        output_depth=target.shape[2],
        encoder_depth=encoder_depth,
        decoder_depth=decoder_depth,
        num_layers=2,
        num_heads=4,
        key_dims=None,
        pos_dims=pos_dims,
        pos_norms=64,
        max_lengths=[source.shape[1], target.shape[1]],
        ff_units=16,
        dropout_rates=0.1,
        pos_sensitive=True,
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
        key_dims=None,
        pos_dims=None,
        pos_norms=64,
        max_lengths=[source.shape[1], target.shape[1]],
        ff_units=16,
        dropout_rates=0.1,
        pos_sensitive=False,
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
