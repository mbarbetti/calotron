import pytest
import tensorflow as tf


@pytest.fixture
def layer():
    from calotron.layers import ModulatedLayerNorm

    ln = ModulatedLayerNorm(axis=-1, epsilon=0.001)
    return ln


###########################################################################


def test_layer_configuration(layer):
    from calotron.layers import ModulatedLayerNorm

    assert isinstance(layer, ModulatedLayerNorm)
    assert isinstance(layer.axis, int)
    assert isinstance(layer.epsilon, float)


def test_layer_use(layer):
    input = tf.keras.Input(shape=(16, 24))
    latent = tf.keras.Input(shape=(24))
    output = layer(input, w=latent)
    assert output.shape == input.shape
