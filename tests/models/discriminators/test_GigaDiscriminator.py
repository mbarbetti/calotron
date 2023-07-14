import pytest
import tensorflow as tf

from calotron.layers.AdminResidual import OUTPUT_CHANGE_SCALES

CHUNK_SIZE = int(1e4)
BATCH_SIZE = 500
ADDITIONAL_DIM = 2

source = tf.random.normal(shape=(CHUNK_SIZE, 8, 5))
target = tf.random.normal(shape=(CHUNK_SIZE, 4, 3))
weight = tf.random.uniform(shape=(CHUNK_SIZE, target.shape[1]))
labels = tf.random.uniform(shape=(CHUNK_SIZE,), minval=0.0, maxval=1.0)
labels = tf.cast(labels > 0.5, target.dtype)


@pytest.fixture
def model():
    from calotron.models.discriminators import GigaDiscriminator

    disc = GigaDiscriminator(
        output_units=1,
        encoder_depth=8,
        decoder_depth=8,
        num_layers=2,
        num_heads=4,
        key_dim=32,
        admin_res_scale="O(n)",
        mlp_units=128,
        dropout_rate=0.1,
        seq_ord_latent_dim=16,
        seq_ord_max_length=max(source.shape[1], target.shape[1]),
        seq_ord_normalization=10_000,
        enable_res_smoothing=True,
        output_activation="sigmoid",
    )
    return disc


###########################################################################


def test_model_configuration(model):
    from calotron.models.discriminators import GigaDiscriminator

    assert isinstance(model, GigaDiscriminator)
    assert isinstance(model.output_units, int)
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
    assert isinstance(model.output_activation, str)
    assert isinstance(model.condition_aware, bool)


@pytest.mark.parametrize("admin_res_scale", OUTPUT_CHANGE_SCALES)
@pytest.mark.parametrize("enable_res_smoothing", [True, False])
@pytest.mark.parametrize("padding_mask", [weight, None])
def test_model_use(admin_res_scale, enable_res_smoothing, padding_mask):
    latent_dim = 8
    if enable_res_smoothing:
        encoder_depth = latent_dim + ADDITIONAL_DIM
        decoder_depth = latent_dim + ADDITIONAL_DIM
    else:
        encoder_depth = latent_dim
        decoder_depth = latent_dim
    from calotron.models.discriminators import GigaDiscriminator

    model = GigaDiscriminator(
        output_units=1,
        encoder_depth=encoder_depth,
        decoder_depth=decoder_depth,
        num_layers=2,
        num_heads=4,
        key_dim=32,
        admin_res_scale=admin_res_scale,
        mlp_units=128,
        dropout_rate=0.1,
        seq_ord_latent_dim=latent_dim,
        seq_ord_max_length=max(source.shape[1], target.shape[1]),
        seq_ord_normalization=10_000,
        enable_res_smoothing=enable_res_smoothing,
        output_activation="sigmoid",
    )
    output = model((source, target), padding_mask=padding_mask)
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
    model.fit(dataset, epochs=1)
