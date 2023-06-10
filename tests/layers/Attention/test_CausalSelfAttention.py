import pytest
import tensorflow as tf


@pytest.fixture
def layer():
    from calotron.layers import CausalSelfAttention

    att = CausalSelfAttention(
        num_heads=8,
        key_dim=64,
        embed_dim=24,
        num_res_layers=5,
        admin_res_scale="O(n)",
        dropout_rate=0.1,
    )
    return att


###########################################################################


def test_layer_configuration(layer):
    from calotron.layers import CausalSelfAttention

    assert isinstance(layer, CausalSelfAttention)
    assert isinstance(layer.num_heads, int)
    assert isinstance(layer.key_dim, int)
    assert isinstance(layer.embed_dim, int)
    assert isinstance(layer.num_res_layers, int)
    assert isinstance(layer.admin_res_scale, str)
    assert isinstance(layer.dropout_rate, float)


def test_layer_use(layer):
    input = tf.random.normal(shape=(100, 32, 24))
    output = layer(input)
    assert output.shape == input.shape
