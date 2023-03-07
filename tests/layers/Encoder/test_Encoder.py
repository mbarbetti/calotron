import pytest
import tensorflow as tf


@pytest.fixture
def layer():
    from calotron.layers import Encoder

    enc = Encoder(
        output_depth=16,
        num_layers=4,
        num_heads=8,
        key_dim=32,
        fnn_units=128,
        dropout_rate=0.1,
        seq_ord_latent_dim=16,
        seq_ord_max_length=512,
        seq_ord_normalization=10_000,
        residual_smoothing=True,
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
    assert isinstance(layer.fnn_units, int)
    assert isinstance(layer.dropout_rate, float)
    assert isinstance(layer.seq_ord_latent_dim, int)
    assert isinstance(layer.seq_ord_max_length, int)
    assert isinstance(layer.seq_ord_normalization, float)
    assert isinstance(layer.residual_smoothing, bool)


@pytest.mark.parametrize("residual_smoothing", [True, False])
def test_layer_use(residual_smoothing):
    latent_dim = 8
    max_length = 32
    if residual_smoothing:
        input_dim = 4
        output_dim = 32  # != 4 (input_dim) + 8 (latent_dim)
    else:
        input_dim = 4
        output_dim = 12  # = 4 (input_dim) + 8 (latent_dim)
    from calotron.layers import Encoder

    layer = Encoder(
        output_depth=output_dim,
        num_layers=4,
        num_heads=8,
        key_dim=32,
        fnn_units=128,
        dropout_rate=0.1,
        seq_ord_latent_dim=latent_dim,
        seq_ord_max_length=max_length,
        seq_ord_normalization=10_000,
        residual_smoothing=residual_smoothing,
    )
    input = tf.random.normal(shape=(100, max_length, input_dim))
    output = layer(input)
    test_shape = list(input.shape)
    test_shape[-1] = layer.output_depth
    assert output.shape == tuple(test_shape)
