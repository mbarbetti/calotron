import pytest
import tensorflow as tf


@pytest.fixture
def layer():
    from calotron.layers import DeepSets

    map = DeepSets(latent_dim=16, num_layers=3, hidden_units=32, dropout_rate=0.1)
    return map


###########################################################################


def test_layer_configuration(layer):
    from calotron.layers.DeepSets import DeepSets

    assert isinstance(layer, DeepSets)
    assert isinstance(layer.latent_dim, int)
    assert isinstance(layer.num_layers, int)
    assert isinstance(layer.hidden_units, int)
    assert isinstance(layer.dropout_rate, float)


@pytest.mark.parametrize("filter", [tf.keras.Input(shape=(8,)), None])
def test_layer_use(layer, filter):
    input = tf.keras.Input(shape=(8, 4))
    output = layer(input, filter=filter)
    assert output.shape[1] == layer.latent_dim
