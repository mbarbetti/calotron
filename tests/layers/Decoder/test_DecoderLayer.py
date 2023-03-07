import pytest
import tensorflow as tf


@pytest.fixture
def layer():
    from calotron.layers.Decoder import DecoderLayer

    dec = DecoderLayer(
        output_depth=16,
        num_heads=8,
        key_dim=32,
        fnn_units=128,
        dropout_rate=0.1,
        residual_smoothing=True,
    )
    return dec


###########################################################################


def test_layer_configuration(layer):
    from calotron.layers.Decoder import DecoderLayer

    assert isinstance(layer, DecoderLayer)
    assert isinstance(layer.output_depth, int)
    assert isinstance(layer.num_heads, int)
    assert isinstance(layer.key_dim, int)
    assert isinstance(layer.fnn_units, int)
    assert isinstance(layer.dropout_rate, float)
    assert isinstance(layer.residual_smoothing, bool)


@pytest.mark.parametrize("residual_smoothing", [True, False])
def test_layer_use(residual_smoothing):
    if residual_smoothing:
        input_dim, output_dim = (8, 16)
    else:
        input_dim, output_dim = (8, 8)
    from calotron.layers.Decoder import DecoderLayer

    layer = DecoderLayer(
        output_depth=output_dim,
        num_heads=8,
        key_dim=32,
        fnn_units=128,
        dropout_rate=0.1,
        residual_smoothing=residual_smoothing,
    )
    source = tf.random.normal(shape=(100, 10, 5))
    target = tf.random.normal(shape=(100, 32, input_dim))
    output = layer(target, source)
    test_shape = list(target.shape)
    test_shape[-1] = layer.output_depth
    assert output.shape == tuple(test_shape)
