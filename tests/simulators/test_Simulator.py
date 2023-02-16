import pytest
import tensorflow as tf

from calotron.models import Transformer

CHUNK_SIZE = int(1e5)
BATCH_SIZE = 100

source = tf.random.normal(shape=(CHUNK_SIZE, 2, 5))
target = tf.random.normal(shape=(CHUNK_SIZE, 4, 10))
model = Transformer(
    output_depth=target.shape[2],
    encoder_depth=8,
    decoder_depth=8,
    num_layers=2,
    num_heads=4,
    key_dim=None,
    ff_units=16,
    dropout_rate=0.1,
)


@pytest.fixture
def simulator():
    from calotron.simulators import Simulator

    sim = Simulator(transformer=model, start_token=target[:BATCH_SIZE, 0, :])
    return sim


###########################################################################


def test_simulator_configuration(simulator):
    from calotron.simulators import Simulator

    assert isinstance(simulator, Simulator)
    assert isinstance(simulator.transformer, Transformer)
    assert isinstance(simulator.start_token, tf.Tensor)


def test_simulator_use(simulator):
    output = simulator(source=source[:BATCH_SIZE], max_length=target.shape[1])
    test_shape = list(target.shape)
    test_shape[0] = BATCH_SIZE
    assert output.shape == tuple(test_shape)
