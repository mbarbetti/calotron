import pytest
import tensorflow as tf


@pytest.fixture
def layer():
    from calotron.layers import SelfAttention

    att = SelfAttention(
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
    from calotron.layers import SelfAttention

    assert isinstance(layer, SelfAttention)
    assert isinstance(layer.num_heads, int)
    assert isinstance(layer.key_dim, int)
    assert isinstance(layer.embed_dim, int)
    assert isinstance(layer.num_res_layers, int)
    assert isinstance(layer.admin_res_scale, str)
    assert isinstance(layer.dropout_rate, float)


@pytest.mark.parametrize("use_causal_mask", [True, False])
def test_layer_use(layer, use_causal_mask):
    input = tf.keras.Input(shape=(16, 24))
    output = layer(input, use_causal_mask=use_causal_mask)
    assert output.shape == input.shape
