import os

import numpy as np
import pytest
import tensorflow as tf

from calotron.models.transformers import Transformer
from calotron.simulators import Simulator

CHUNK_SIZE = int(1e4)
BATCH_SIZE = 100

here = os.path.dirname(__file__)
export_dir = f"{here}/tmp/simulator"

source = tf.random.normal(shape=(CHUNK_SIZE, 8, 3))
target = tf.random.normal(shape=(CHUNK_SIZE, 4, 3))
dataset = tf.data.Dataset.from_tensor_slices(source[:BATCH_SIZE]).batch(
    10, drop_remainder=True
)

model = Transformer(
    output_depth=target.shape[2],
    encoder_depth=8,
    decoder_depth=8,
    num_layers=2,
    num_heads=4,
    key_dims=32,
    mlp_units=128,
    dropout_rates=0.1,
    seq_ord_latent_dims=16,
    seq_ord_max_lengths=[source.shape[1], target.shape[1]],
    seq_ord_normalizations=10_000,
    residual_smoothing=True,
    output_activations="relu",
    start_token_initializer="ones",
)

simulator = Simulator(transformer=model, start_token=[0, 0, 1])


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
    output, attn_weights = export_simulator(dataset)
    test_shape = list(target.shape)
    test_shape[0] = BATCH_SIZE
    assert output.shape == tuple(test_shape)
    test_shape = list()
    test_shape.append(BATCH_SIZE)
    test_shape.append(model.num_heads[1])
    test_shape.append(target.shape[1])
    test_shape.append(source.shape[1])
    assert attn_weights.shape == tuple(test_shape)
    tf.saved_model.save(export_simulator, export_dir=export_dir)
    reloaded = tf.saved_model.load(export_dir)
    output_reloaded, attn_weights_reloaded = reloaded(dataset)
    comparison = output.numpy() == output_reloaded.numpy()
    assert comparison.all()
