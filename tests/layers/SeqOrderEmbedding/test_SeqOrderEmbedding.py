import pytest
import tensorflow as tf


@pytest.fixture
def layer():
    from calotron.layers import SeqOrderEmbedding

    seq_order = SeqOrderEmbedding(latent_dim=8, max_length=512, normalization=10_000)
    return seq_order


###########################################################################


def test_layer_configuration(layer):
    from calotron.layers import SeqOrderEmbedding

    assert isinstance(layer, SeqOrderEmbedding)
    assert isinstance(layer.latent_dim, int)
    assert isinstance(layer.max_length, int)
    assert isinstance(layer.normalization, float)


def test_layer_use(layer):
    input = tf.keras.Input(shape=(16, 5))
    output = layer(input)
    test_shape = list(input.shape)
    test_shape[-1] = layer.latent_dim
    assert output.shape == tuple(test_shape)
