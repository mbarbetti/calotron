import pytest
import tensorflow as tf

###########################################################################


@pytest.mark.parametrize("key_dim", [None, 64])
def test_layer_configuration(key_dim):
    from calotron.layers.Encoder import EncoderLayer

    layer = EncoderLayer(
        encoder_depth=16,
        num_heads=8,
        key_dim=key_dim,
        ff_units=128,
        dropout_rate=0.1,
        residual_smoothing=True,
    )
    assert isinstance(layer, EncoderLayer)
    assert isinstance(layer.encoder_depth, int)
    assert isinstance(layer.num_heads, int)
    if layer.key_dim is not None:
        assert isinstance(layer.key_dim, int)
    assert isinstance(layer.ff_units, int)
    assert isinstance(layer.dropout_rate, float)
    assert isinstance(layer.residual_smoothing, bool)


@pytest.mark.parametrize("key_dim", [None, 64])
@pytest.mark.parametrize("residual_smoothing", [True, False])
def test_layer_use(key_dim, residual_smoothing):
    if residual_smoothing:
        input_dim, output_dim = (10, 16)
    else:
        input_dim, output_dim = (10, 10)
    from calotron.layers.Encoder import EncoderLayer

    layer = EncoderLayer(
        encoder_depth=output_dim,
        num_heads=8,
        key_dim=key_dim,
        ff_units=128,
        dropout_rate=0.1,
        residual_smoothing=residual_smoothing,
    )
    input = tf.random.normal(shape=(100, 32, input_dim))
    output = layer(input)
    test_shape = list(input.shape)
    test_shape[-1] = layer.encoder_depth
    assert output.shape == tuple(test_shape)
