import os

import pytest
import tensorflow as tf

from calotron.layers.AdminResidual import OUTPUT_CHANGE_SCALES
from calotron.models.transformers.Transformer import START_TOKEN_INITIALIZERS

CHUNK_SIZE = int(1e4)
BATCH_SIZE = 500
ADDITIONAL_DIM = 2

here = os.path.dirname(__file__)
encoder_dir = f"{here}/../players/tmp/encoder"

source = tf.random.normal(shape=(CHUNK_SIZE, 8, 5))
target = tf.random.normal(shape=(CHUNK_SIZE, 4, 3))


@pytest.fixture
def model():
    from calotron.models.transformers import OptionalTransformer

    trans = OptionalTransformer(
        output_depth=target.shape[2],
        encoder_options={
            "output_depth": 16,
            "num_layers": 3,
            "num_heads": 2,
            "key_dim": 32,
            "admin_res_scale": "O(n)",
            "mlp_units": 128,
            "dropout_rate": 0.1,
            "seq_ord_latent_dim": 24,
            "seq_ord_max_length": source.shape[1],
            "seq_ord_normalization": 10_000,
            "enable_res_smoothing": True,
        },
        decoder_options={
            "output_depth": 8,
            "num_layers": 2,
            "num_heads": 4,
            "key_dim": 32,
            "admin_res_scale": "O(n)",
            "mlp_units": 128,
            "dropout_rate": 0.1,
            "seq_ord_latent_dim": 16,
            "seq_ord_max_length": target.shape[1],
            "seq_ord_normalization": 10_000,
            "enable_res_smoothing": True,
        },
        output_activations="linear",
        start_token_initializer="ones",
        pretrained_encoder_dir=None,
        additional_encoder_layers=None,
    )
    return trans


###########################################################################


def test_model_configuration(model):
    from calotron.models.transformers import OptionalTransformer

    assert isinstance(model, OptionalTransformer)
    assert isinstance(model.output_depth, int)
    assert isinstance(model.encoder_output_depth, int)
    assert isinstance(model.decoder_output_depth, int)
    assert isinstance(model.encoder_num_layers, int)
    assert isinstance(model.decoder_num_layers, int)
    assert isinstance(model.encoder_num_heads, int)
    assert isinstance(model.decoder_num_heads, int)
    assert isinstance(model.encoder_key_dim, int)
    assert isinstance(model.decoder_key_dim, int)
    assert isinstance(model.encoder_admin_res_scale, str)
    assert isinstance(model.decoder_admin_res_scale, str)
    assert isinstance(model.encoder_mlp_units, int)
    assert isinstance(model.decoder_mlp_units, int)
    assert isinstance(model.encoder_dropout_rate, float)
    assert isinstance(model.decoder_dropout_rate, float)
    assert isinstance(model.encoder_seq_ord_latent_dim, int)
    assert isinstance(model.decoder_seq_ord_latent_dim, int)
    assert isinstance(model.encoder_seq_ord_max_length, int)
    assert isinstance(model.decoder_seq_ord_max_length, int)
    assert isinstance(model.encoder_seq_ord_normalization, float)
    assert isinstance(model.decoder_seq_ord_normalization, float)
    assert isinstance(model.enable_res_smoothing, bool)
    # assert isinstance(model.output_activations, str)
    assert isinstance(model.start_token_initializer, str)
    # assert isinstance(model.pretrained_encoder_dir, str)
    assert isinstance(model.additional_encoder_layers, int)


@pytest.mark.parametrize("admin_res_scale", OUTPUT_CHANGE_SCALES)
@pytest.mark.parametrize("enable_res_smoothing", [True, False])
@pytest.mark.parametrize("output_activations", ["relu", None])
@pytest.mark.parametrize("pretrained_encoder_dir", [encoder_dir, None])
def test_model_use(
    admin_res_scale, enable_res_smoothing, output_activations, pretrained_encoder_dir
):
    encoder_latent_dim = 8
    decoder_latent_dim = 9
    if enable_res_smoothing:
        encoder_depth = encoder_latent_dim + ADDITIONAL_DIM
        decoder_depth = decoder_latent_dim + ADDITIONAL_DIM
        if pretrained_encoder_dir is not None:
            export_dir = f"{pretrained_encoder_dir}_with_smoothing"
        else:
            export_dir = None
    else:
        encoder_depth = encoder_latent_dim
        decoder_depth = decoder_latent_dim
        if pretrained_encoder_dir is not None:
            export_dir = f"{pretrained_encoder_dir}_no_smoothing"
        else:
            export_dir = None
    from calotron.models.transformers import OptionalTransformer

    model = OptionalTransformer(
        output_depth=target.shape[2],
        encoder_options={
            "output_depth": encoder_depth,
            "num_layers": 3,
            "num_heads": 2,
            "key_dim": 32,
            "admin_res_scale": admin_res_scale,
            "mlp_units": 128,
            "dropout_rate": 0.1,
            "seq_ord_latent_dim": encoder_latent_dim,
            "seq_ord_max_length": source.shape[1],
            "seq_ord_normalization": 10_000,
            "enable_res_smoothing": enable_res_smoothing,
        },
        decoder_options={
            "output_depth": decoder_depth,
            "num_layers": 2,
            "num_heads": 4,
            "key_dim": 32,
            "admin_res_scale": admin_res_scale,
            "mlp_units": 128,
            "dropout_rate": 0.1,
            "seq_ord_latent_dim": decoder_latent_dim,
            "seq_ord_max_length": target.shape[1],
            "seq_ord_normalization": 10_000,
            "enable_res_smoothing": enable_res_smoothing,
        },
        output_activations=output_activations,
        start_token_initializer="ones",
        pretrained_encoder_dir=export_dir,
        additional_encoder_layers=None,
    )
    output = model((source, target))
    model.summary()
    test_shape = list(target.shape)
    test_shape[-1] = model.output_depth
    assert output.shape == tuple(test_shape)


@pytest.mark.parametrize(
    "target",
    [
        tf.random.normal(shape=(CHUNK_SIZE, 4, 3)),
        tf.random.normal(shape=(CHUNK_SIZE, 8, 3)),
        tf.random.normal(shape=(CHUNK_SIZE, 12, 3)),
        tf.random.normal(shape=(CHUNK_SIZE, 4, 6)),
    ],
)
def test_model_baseline(target):
    from calotron.models.transformers import OptionalTransformer

    model = OptionalTransformer(
        output_depth=target.shape[2],
        encoder_options={
            "output_depth": 16,
            "num_layers": 3,
            "num_heads": 2,
            "key_dim": 32,
            "admin_res_scale": "O(n)",
            "mlp_units": 128,
            "dropout_rate": 0.1,
            "seq_ord_latent_dim": 24,
            "seq_ord_max_length": source.shape[1],
            "seq_ord_normalization": 10_000,
            "enable_res_smoothing": True,
        },
        decoder_options={
            "output_depth": 8,
            "num_layers": 2,
            "num_heads": 4,
            "key_dim": 32,
            "admin_res_scale": "O(n)",
            "mlp_units": 128,
            "dropout_rate": 0.1,
            "seq_ord_latent_dim": 16,
            "seq_ord_max_length": target.shape[1],
            "seq_ord_normalization": 10_000,
            "enable_res_smoothing": True,
        },
        output_activations="linear",
        start_token_initializer="ones",
        pretrained_encoder_dir=None,
        additional_encoder_layers=None,
    )
    output = model((source, target))
    model.summary()
    test_shape = list(target.shape)
    test_shape[-1] = model.output_depth
    assert output.shape == tuple(test_shape)


@pytest.mark.parametrize("start_token_initializer", START_TOKEN_INITIALIZERS)
def test_model_start_token(start_token_initializer):
    from calotron.models.transformers import OptionalTransformer

    model = OptionalTransformer(
        output_depth=target.shape[2],
        encoder_options={
            "output_depth": 16,
            "num_layers": 3,
            "num_heads": 2,
            "key_dim": 32,
            "admin_res_scale": "O(n)",
            "mlp_units": 128,
            "dropout_rate": 0.1,
            "seq_ord_latent_dim": 24,
            "seq_ord_max_length": source.shape[1],
            "seq_ord_normalization": 10_000,
            "enable_res_smoothing": True,
        },
        decoder_options={
            "output_depth": 8,
            "num_layers": 2,
            "num_heads": 4,
            "key_dim": 32,
            "admin_res_scale": "O(n)",
            "mlp_units": 128,
            "dropout_rate": 0.1,
            "seq_ord_latent_dim": 16,
            "seq_ord_max_length": target.shape[1],
            "seq_ord_normalization": 10_000,
            "enable_res_smoothing": True,
        },
        output_activations="linear",
        start_token_initializer=start_token_initializer,
        pretrained_encoder_dir=None,
        additional_encoder_layers=None,
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
    model.fit(dataset, epochs=1)
