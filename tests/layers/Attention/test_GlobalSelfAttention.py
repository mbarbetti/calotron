import pytest
import tensorflow as tf


@pytest.fixture
def layer():
    from calotron.layers import GlobalSelfAttention

    att = GlobalSelfAttention(num_heads=8, key_dim=64)
    return att


###########################################################################


def test_layer_configuration(layer):
    from calotron.layers import GlobalSelfAttention

    assert isinstance(layer, GlobalSelfAttention)


def test_layer_use(layer):
    input = tf.keras.Input(shape=[16, 5])
    output = layer(input)
    assert output.shape == input.shape
