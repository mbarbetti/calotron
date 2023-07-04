import pytest
import tensorflow as tf

from calotron.layers.AdminResidual import OUTPUT_CHANGE_SCALES


@pytest.fixture
def layer():
    from calotron.layers import AdminResidual

    res = AdminResidual(embed_dim=24, num_res_layers=5, output_change_scale="O(logn)")
    return res


###########################################################################


def test_layer_configuration(layer):
    from calotron.layers import AdminResidual

    assert isinstance(layer, AdminResidual)
    assert isinstance(layer.embed_dim, int)
    assert isinstance(layer.num_res_layers, int)
    assert isinstance(layer.output_change_scale, str)


@pytest.mark.parametrize("output_change_scale", OUTPUT_CHANGE_SCALES)
def test_layer_use(output_change_scale):
    from calotron.layers import AdminResidual

    layer = AdminResidual(
        embed_dim=24, num_res_layers=5, output_change_scale=output_change_scale
    )
    x = tf.keras.Input(shape=(16, 24))
    f_x = tf.keras.Input(shape=(16, 24))
    output = layer([x, f_x])
    test_shape = list(x.shape)
    test_shape[-1] = layer.embed_dim
    assert output.shape == tuple(test_shape)
