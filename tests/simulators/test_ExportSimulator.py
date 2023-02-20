import os

import numpy as np
import pytest
import tensorflow as tf

from calotron.models import Transformer
from calotron.simulators import Simulator

CHUNK_SIZE = int(1e4)
BATCH_SIZE = 100

here = os.path.dirname(__file__)
export_dir = f"{here}/tmp/simulator"

source = tf.random.normal(shape=(CHUNK_SIZE, 16, 3))
target = tf.random.normal(shape=(CHUNK_SIZE, 32, 9))

model = Transformer(
    output_depth=target.shape[2],
    encoder_depth=8,
    decoder_depth=8,
    num_layers=2,
    num_heads=4,
    key_dim=None,
    encoder_pos_dim=None,
    decoder_pos_dim=None,
    encoder_pos_normalization=64,
    decoder_pos_normalization=64,
    encoder_max_length=source.shape[1],
    decoder_max_length=target.shape[1],
    ff_units=16,
    dropout_rate=0.1,
    pos_sensitive=False,
    residual_smoothing=True,
    output_activations="relu",
    start_token_initializer="zeros",
)

simulator = Simulator(
    transformer=model, start_token=model.get_start_token(target[:BATCH_SIZE])
)


@pytest.fixture
def export_simulator():
    from calotron.simulators import ExportSimulator

    sim = ExportSimulator(simulator=simulator, max_length=target.shape[1])
    return sim


###########################################################################


def test_export_simulator_configuration(export_simulator):
    from calotron.simulators import ExportSimulator

    assert isinstance(export_simulator, ExportSimulator)
    assert isinstance(export_simulator.simulator, Simulator)
    assert isinstance(export_simulator.max_length, int)


def test_export_use(export_simulator):
    output = export_simulator(source=source[:BATCH_SIZE])
    test_shape = list(target.shape)
    test_shape[0] = BATCH_SIZE
    assert output.shape == tuple(test_shape)
    tf.saved_model.save(export_simulator, export_dir=export_dir)
    reloaded = tf.saved_model.load(export_dir)
    output_reloaded = reloaded(source=source[:BATCH_SIZE])
    comparison = output.numpy() == output_reloaded.numpy()
    assert comparison.all()
