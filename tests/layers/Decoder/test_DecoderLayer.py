import pytest
import tensorflow as tf


@pytest.fixture
def layer():
    from calotron.layers.Decoder import DecoderLayer

    dec = DecoderLayer(
        output_depth=24,
        num_heads=8,
        key_dim=32,
        num_res_layers=5,
        admin_res_scale="O(n)",
        mlp_units=128,
        dropout_rate=0.1,
    )
    return dec


###########################################################################


def test_layer_configuration(layer):
    from calotron.layers.Decoder import DecoderLayer

    assert isinstance(layer, DecoderLayer)
    assert isinstance(layer.output_depth, int)
    assert isinstance(layer.num_heads, int)
    assert isinstance(layer.key_dim, int)
    assert isinstance(layer.num_res_layers, int)
    assert isinstance(layer.admin_res_scale, str)
    assert isinstance(layer.mlp_units, int)
    assert isinstance(layer.dropout_rate, float)


def test_layer_use(layer):
    source = tf.random.normal(shape=(100, 16, 24))
    target = tf.random.normal(shape=(100, 8, 24))
    output = layer(target, source)
    test_shape = list(target.shape)
    test_shape[-1] = layer.output_depth
    assert output.shape == tuple(test_shape)
