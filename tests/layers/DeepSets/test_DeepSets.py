import pytest
import tensorflow as tf


@pytest.fixture
def layer():
    from calotron.layers import DeepSets

    map = DeepSets(latent_dim=16, num_layers=1, hidden_units=32, dropout_rate=0.1)
    return map


###########################################################################


def test_layer_configuration(layer):
    from calotron.layers.DeepSets import DeepSets

    assert isinstance(layer, DeepSets)
    assert isinstance(layer.latent_dim, int)
    assert isinstance(layer.num_layers, int)
    assert isinstance(layer.hidden_units, int)
    assert isinstance(layer.dropout_rate, float)


@pytest.mark.parametrize("padding_mask", [tf.keras.Input(shape=(8,)), None])
def test_layer_use(layer, padding_mask):
    input = tf.keras.Input(shape=(8, 4))
    output = layer(input, padding_mask=padding_mask)
    assert output.shape[1] == layer.latent_dim
