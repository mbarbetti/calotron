import pytest
import tensorflow as tf


@pytest.fixture
def layer():
    from calotron.layers import MappingNet

    map = MappingNet(
        output_dim=32, latent_dim=64, num_layers=4, hidden_units=128, dropout_rate=0.1
    )
    return map


###########################################################################


def test_layer_configuration(layer):
    from calotron.layers import MappingNet

    assert isinstance(layer, MappingNet)
    assert isinstance(layer.output_dim, int)
    assert isinstance(layer.latent_dim, int)
    assert isinstance(layer.num_layers, int)
    assert isinstance(layer.hidden_units, int)
    assert isinstance(layer.dropout_rate, float)


def test_layer_use(layer):
    input = tf.keras.Input(shape=(32,))
    output = layer(input)
    test_shape = list(input.shape)
    test_shape[-1] = layer.output_dim
    assert output.shape == tuple(test_shape)
