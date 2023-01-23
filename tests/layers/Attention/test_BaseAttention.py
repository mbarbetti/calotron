import pytest
import tensorflow as tf


@pytest.fixture
def layer():
    from calotron.layers.Attention import BaseAttention

    att = BaseAttention(num_heads=8, key_dim=64)
    return att


###########################################################################


def test_layer_configuration(layer):
    from calotron.layers.Attention import BaseAttention

    assert isinstance(layer, BaseAttention)
