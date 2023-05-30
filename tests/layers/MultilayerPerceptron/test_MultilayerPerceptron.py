import pytest
import tensorflow as tf


@pytest.fixture
def layer():
    from calotron.layers import MultilayerPerceptron

    mlp = MultilayerPerceptron(output_units=32, hidden_units=128, dropout_rate=0.1)
    return mlp


###########################################################################


def test_layer_configuration(layer):
    from calotron.layers import MultilayerPerceptron

    assert isinstance(layer, MultilayerPerceptron)
    assert isinstance(layer.output_units, int)
    assert isinstance(layer.hidden_units, int)
    assert isinstance(layer.dropout_rate, float)


def test_layer_use(layer):
    input = tf.keras.Input(shape=(16, 32))
    output = layer(input)
    test_shape = list(input.shape)
    test_shape[-1] = layer.output_units
    assert output.shape == tuple(test_shape)
