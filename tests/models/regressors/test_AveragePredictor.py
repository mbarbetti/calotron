import pytest
import tensorflow as tf

from calotron.layers.AdminResidual import OUTPUT_CHANGE_SCALES

CHUNK_SIZE = int(1e4)
BATCH_SIZE = 500
ADDITIONAL_DIM = 2

source = tf.random.normal(shape=(CHUNK_SIZE, 8, 5))
target = tf.random.normal(shape=(CHUNK_SIZE, 4, 3))


@pytest.fixture
def model():
    from calotron.models.regressors import AveragePredictor

    pred = AveragePredictor(
        output_units=target.shape[2],
        encoder_depth=8,
        num_layers=2,
        num_heads=4,
        key_dim=32,
        admin_res_scale="O(n)",
        mlp_units=128,
        dropout_rate=0.1,
        seq_ord_latent_dim=16,
        seq_ord_max_length=source.shape[1],
        seq_ord_normalization=10_000,
        enable_res_smoothing=True,
        output_activation="linear",
    )
    return pred


###########################################################################


def test_model_configuration(model):
    from calotron.models.players import Encoder
    from calotron.models.regressors import AveragePredictor

    assert isinstance(model, AveragePredictor)
    assert isinstance(model.output_units, int)
    assert isinstance(model.encoder_depth, int)
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
    # assert isinstance(model.output_activation, str)
    assert isinstance(model.encoder, Encoder)


@pytest.mark.parametrize("admin_res_scale", OUTPUT_CHANGE_SCALES)
@pytest.mark.parametrize("enable_res_smoothing", [True, False])
@pytest.mark.parametrize("output_activation", ["relu", None])
def test_model_use(admin_res_scale, enable_res_smoothing, output_activation):
    latent_dim = 8
    encoder_depth = latent_dim + ADDITIONAL_DIM if enable_res_smoothing else latent_dim
    from calotron.models.regressors import AveragePredictor

    model = AveragePredictor(
        output_units=target.shape[2],
        encoder_depth=encoder_depth,
        num_layers=2,
        num_heads=4,
        key_dim=32,
        admin_res_scale=admin_res_scale,
        mlp_units=128,
        dropout_rate=0.1,
        seq_ord_latent_dim=latent_dim,
        seq_ord_max_length=source.shape[1],
        seq_ord_normalization=10_000,
        enable_res_smoothing=enable_res_smoothing,
        output_activation=output_activation,
    )
    output = model(source)
    model.summary()
    test_shape = [source.shape[0]]
    test_shape.append(model.output_units)
    assert output.shape == tuple(test_shape)


def test_model_train(model):
    reduced_target = tf.reduce_mean(target, axis=1)
    dataset = (
        tf.data.Dataset.from_tensor_slices((source, reduced_target))
        .batch(batch_size=BATCH_SIZE, drop_remainder=True)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    mse = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=adam, loss=mse)
    model.fit(dataset, epochs=1)
