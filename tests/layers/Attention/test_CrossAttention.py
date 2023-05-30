import pytest
import tensorflow as tf


@pytest.fixture
def layer():
    from calotron.layers import CrossAttention

    att = CrossAttention(num_heads=8, key_dim=64)
    return att


###########################################################################


def test_layer_configuration(layer):
    from calotron.layers import CrossAttention

    assert isinstance(layer, CrossAttention)


def test_layer_use(layer):
    source = tf.keras.Input(shape=(16, 5))
    target = tf.keras.Input(shape=(32, 10))
    output = layer(target, source)
    assert output.shape == target.shape
