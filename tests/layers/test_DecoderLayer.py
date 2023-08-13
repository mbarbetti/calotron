import pytest
import tensorflow as tf


@pytest.fixture
def layer():
    from calotron.layers import DecoderLayer

    dec = DecoderLayer(
        output_depth=12,
        num_heads=8,
        key_dim=32,
        num_res_layers=5,
        admin_res_scale="O(n)",
        mlp_units=128,
        dropout_rate=0.1,
        autoregressive_mode=True,
    )
    return dec


###########################################################################


def test_layer_configuration(layer):
    from calotron.layers import DecoderLayer

    assert isinstance(layer, DecoderLayer)
    assert isinstance(layer.output_depth, int)
    assert isinstance(layer.num_heads, int)
    assert isinstance(layer.key_dim, int)
    assert isinstance(layer.num_res_layers, int)
    assert isinstance(layer.admin_res_scale, str)
    assert isinstance(layer.mlp_units, int)
    assert isinstance(layer.dropout_rate, float)
    assert isinstance(layer.autoregressive_mode, bool)


@pytest.mark.parametrize("autoregressive_mode", [True, False])
def test_layer_use(autoregressive_mode):
    from calotron.layers import DecoderLayer

    layer = DecoderLayer(
        output_depth=12,
        num_heads=8,
        key_dim=32,
        num_res_layers=5,
        admin_res_scale="O(n)",
        mlp_units=128,
        dropout_rate=0.1,
        autoregressive_mode=autoregressive_mode,
    )
    source = tf.random.normal(shape=(100, 16, 24))
    target = tf.random.normal(shape=(100, 8, 12))
    output = layer(target, condition=source)
    test_shape = list(target.shape)
    test_shape[-1] = layer.output_depth
    assert output.shape == tuple(test_shape)
