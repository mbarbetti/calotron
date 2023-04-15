import pytest
import tensorflow as tf

ADDITIONAL_DIM = 2


@pytest.fixture
def layer():
    from calotron.layers import Encoder

    enc = Encoder(
        output_depth=16,
        num_layers=4,
        num_heads=8,
        key_dim=32,
        mlp_units=128,
        dropout_rate=0.1,
        seq_ord_latent_dim=16,
        seq_ord_max_length=512,
        seq_ord_normalization=10_000,
        enable_residual_smoothing=True,
    )
    return enc


###########################################################################


def test_layer_configuration(layer):
    from calotron.layers import Encoder

    assert isinstance(layer, Encoder)
    assert isinstance(layer.output_depth, int)
    assert isinstance(layer.num_layers, int)
    assert isinstance(layer.num_heads, int)
    assert isinstance(layer.key_dim, int)
    assert isinstance(layer.mlp_units, int)
    assert isinstance(layer.dropout_rate, float)
    assert isinstance(layer.seq_ord_latent_dim, int)
    assert isinstance(layer.seq_ord_max_length, int)
    assert isinstance(layer.seq_ord_normalization, float)
    assert isinstance(layer.enable_residual_smoothing, bool)


@pytest.mark.parametrize("enable_residual_smoothing", [True, False])
def test_layer_use(enable_residual_smoothing):
    input_dim = 4
    latent_dim = 8
    max_length = 32
    if enable_residual_smoothing:
        output_dim = latent_dim + ADDITIONAL_DIM
    else:
        output_dim = latent_dim
    from calotron.layers import Encoder

    layer = Encoder(
        output_depth=output_dim,
        num_layers=4,
        num_heads=8,
        key_dim=32,
        mlp_units=128,
        dropout_rate=0.1,
        seq_ord_latent_dim=latent_dim,
        seq_ord_max_length=max_length,
        seq_ord_normalization=10_000,
        enable_residual_smoothing=enable_residual_smoothing,
    )
    input = tf.random.normal(shape=(100, max_length, input_dim))
    output = layer(input)
    test_shape = list(input.shape)
    test_shape[-1] = layer.output_depth
    assert output.shape == tuple(test_shape)
