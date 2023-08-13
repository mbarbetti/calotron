import os

import pytest
import tensorflow as tf

from calotron.layers.AdminResidual import OUTPUT_CHANGE_SCALES

CHUNK_SIZE = int(1e4)
BATCH_SIZE = 500
ADDITIONAL_DIM = 2

here = os.path.dirname(__file__)
encoder_dir = f"{here}/tmp/encoder"

source = tf.random.normal(shape=(CHUNK_SIZE, 8, 5))
target = tf.random.normal(shape=(CHUNK_SIZE, 8, 10))


@pytest.fixture
def model():
    from calotron.models.players import PretrainedEncoder

    enc = PretrainedEncoder(
        output_depth=target.shape[-1],
        num_layers=4,
        num_heads=8,
        key_dim=32,
        admin_res_scale="O(n)",
        mlp_units=128,
        dropout_rate=0.1,
        seq_ord_latent_dim=16,
        seq_ord_max_length=512,
        seq_ord_normalization=10_000,
        enable_res_smoothing=True,
        pretrained_model_dir=f"{encoder_dir}_with_smoothing",
    )
    return enc


###########################################################################


def test_model_configuration(model):
    from calotron.models.players import PretrainedEncoder

    assert isinstance(model, PretrainedEncoder)
    assert isinstance(model.output_depth, int)
    assert isinstance(model.num_layers, int)
    assert isinstance(model.num_heads, int)
    assert isinstance(model.key_dim, int)
    assert isinstance(model.admin_res_scale, str)
    assert isinstance(model.mlp_units, int)
    assert isinstance(model.dropout_rate, float)
    assert isinstance(model.seq_ord_latent_dim, int)
    assert isinstance(model.seq_ord_max_length, int)
    assert isinstance(model.seq_ord_normalization, float)
    assert isinstance(model.enable_res_smoothing, bool)
    assert isinstance(model.pretrained_model_dir, str)


@pytest.mark.parametrize("admin_res_scale", OUTPUT_CHANGE_SCALES)
@pytest.mark.parametrize("enable_res_smoothing", [True, False])
@pytest.mark.parametrize("pretrained_model_dir", [encoder_dir, None])
def test_model_use(admin_res_scale, enable_res_smoothing, pretrained_model_dir):
    latent_dim = 8
    if enable_res_smoothing:
        output_dim = latent_dim + ADDITIONAL_DIM
        if pretrained_model_dir is not None:
            export_dir = f"{pretrained_model_dir}_with_smoothing"
        else:
            export_dir = None
    else:
        output_dim = latent_dim
        if pretrained_model_dir is not None:
            export_dir = f"{pretrained_model_dir}_no_smoothing"
        else:
            export_dir = None
    from calotron.models.players import PretrainedEncoder

    tf.print(pretrained_model_dir)

    model = PretrainedEncoder(
        output_depth=output_dim,
        num_layers=4,
        num_heads=8,
        key_dim=32,
        admin_res_scale=admin_res_scale,
        mlp_units=128,
        dropout_rate=0.1,
        seq_ord_latent_dim=latent_dim,
        seq_ord_max_length=source.shape[1],
        seq_ord_normalization=10_000,
        enable_res_smoothing=enable_res_smoothing,
        pretrained_model_dir=export_dir,
    )
    output = model(source)
    model.summary()
    test_shape = list(source.shape)
    test_shape[-1] = model.output_depth
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
