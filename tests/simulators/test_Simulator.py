import numpy as np
import pytest
import tensorflow as tf

from calotron.models import Transformer

CHUNK_SIZE = int(1e4)
BATCH_SIZE = 100

source = tf.random.normal(shape=(CHUNK_SIZE, 8, 3))
target = tf.random.normal(shape=(CHUNK_SIZE, 4, 3))

model = Transformer(
    output_depth=target.shape[2],
    encoder_depth=8,
    decoder_depth=8,
    num_layers=2,
    num_heads=4,
    key_dims=32,
    fnn_units=128,
    dropout_rates=0.1,
    seq_ord_latent_dims=16,
    seq_ord_max_lengths=[source.shape[1], target.shape[1]],
    seq_ord_normalizations=10_000,
    residual_smoothing=True,
    output_activations="relu",
    start_token_initializer="ones",
)

start_token_tf = model.get_start_token(target[:BATCH_SIZE])
start_token_np = np.array([0, 0, 1])


@pytest.fixture
def simulator():
    from calotron.simulators import Simulator

    sim = Simulator(transformer=model, start_token=start_token_tf)
    return sim


###########################################################################


def test_simulator_configuration(simulator):
    from calotron.simulators import Simulator

    assert isinstance(simulator, Simulator)
    assert isinstance(simulator.transformer, Transformer)
    assert isinstance(simulator.start_token, np.ndarray)


@pytest.mark.parametrize("start_token", [start_token_tf, start_token_np])
@pytest.mark.parametrize("source", [source, source.numpy()])
def test_simulator_use(start_token, source):
    from calotron.simulators import Simulator

    sim = Simulator(transformer=model, start_token=start_token)
    output = sim(source=source[:BATCH_SIZE], max_length=target.shape[1])
    test_shape = list(target.shape)
    test_shape[0] = BATCH_SIZE
    assert output.shape == tuple(test_shape)
