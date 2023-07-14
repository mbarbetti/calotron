import pytest
import tensorflow as tf


@pytest.fixture
def layer():
    from calotron.layers import ConvDeepSets

    map = ConvDeepSets(
        latent_dim=16,
        num_conv_layers=1,
        filters=8,
        kernel_size=2,
        strides=1,
        dropout_rate=0.1,
    )
    return map


###########################################################################


def test_layer_configuration(layer):
    from calotron.layers.ConvDeepSets import ConvDeepSets

    assert isinstance(layer, ConvDeepSets)
    assert isinstance(layer.latent_dim, int)
    assert isinstance(layer.num_conv_layers, int)
    assert isinstance(layer.filters, int)
    assert isinstance(layer.kernel_size, int)
    assert isinstance(layer.strides, int)
    assert isinstance(layer.dropout_rate, float)


@pytest.mark.parametrize("padding_mask", [tf.keras.Input(shape=(8,)), None])
def test_layer_use(layer, padding_mask):
    input = tf.keras.Input(shape=(8, 4))
    output = layer(input, padding_mask=padding_mask)
    assert output.shape[1] == layer.latent_dim
