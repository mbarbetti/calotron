import pytest
import tensorflow as tf

from calotron.models.discriminators import Discriminator
from calotron.models.transformers import Transformer

CHUNK_SIZE = int(1e4)
source = tf.random.normal(shape=(CHUNK_SIZE, 8, 5))
target = tf.random.normal(shape=(CHUNK_SIZE, 4, 3))
weight = tf.random.uniform(shape=(CHUNK_SIZE, target.shape[1], 1))

transf = Transformer(
    output_depth=target.shape[2],
    encoder_depth=8,
    decoder_depth=8,
    num_layers=2,
    num_heads=4,
    key_dims=32,
    mlp_units=16,
    dropout_rates=0.1,
    seq_ord_latent_dims=16,
    seq_ord_max_lengths=[source.shape[1], target.shape[1]],
    seq_ord_normalizations=10_000,
    enable_residual_smoothing=True,
    output_activations="relu",
    start_token_initializer="ones",
)

disc = Discriminator(
    latent_dim=8,
    output_units=1,
    output_activation=None,
    deepsets_num_layers=2,
    deepsets_hidden_units=32,
    dropout_rate=0.1,
)


@pytest.fixture
def loss():
    from calotron.losses import BinaryCrossentropy

    loss_ = BinaryCrossentropy(
        injected_noise_stddev=0.01, from_logits=False, label_smoothing=0.1, ignore_padding=False,
    )
    return loss_


###########################################################################


def test_loss_configuration(loss):
    from calotron.losses import BinaryCrossentropy

    assert isinstance(loss, BinaryCrossentropy)
    assert isinstance(loss.injected_noise_stddev, float)
    assert isinstance(loss.from_logits, bool)
    assert isinstance(loss.label_smoothing, float)
    assert isinstance(loss.ignore_padding, bool)
    assert isinstance(loss.name, str)


@pytest.mark.parametrize("from_logits", [False, True])
def test_loss_use_no_weights(from_logits):
    from calotron.losses import BinaryCrossentropy

    loss = BinaryCrossentropy(
        injected_noise_stddev=0.01, from_logits=from_logits, label_smoothing=0.1, ignore_padding=False
    )
    if from_logits:
        disc._seq += [tf.keras.layers.Dense(1, activation="tanh")]
    else:
        disc._seq += [tf.keras.layers.Dense(1, activation="sigmoid")]
    out = loss.transformer_loss(
        transformer=transf,
        discriminator=disc,
        source=source,
        target=target,
        sample_weight=None,
        training=False,
    )
    assert out.numpy()
    out = loss.discriminator_loss(
        transformer=transf,
        discriminator=disc,
        source=source,
        target=target,
        sample_weight=None,
        training=False,
    )
    assert out.numpy()


@pytest.mark.parametrize("from_logits", [False, True])
def test_loss_use_with_weights(from_logits):
    from calotron.losses import BinaryCrossentropy

    loss = BinaryCrossentropy(
        injected_noise_stddev=0.01, from_logits=from_logits, label_smoothing=0.1, ignore_padding=False
    )
    if from_logits:
        disc._seq += [tf.keras.layers.Dense(1, activation="tanh")]
    else:
        disc._seq += [tf.keras.layers.Dense(1, activation="sigmoid")]
    out = loss.transformer_loss(
        transformer=transf,
        discriminator=disc,
        source=source,
        target=target,
        sample_weight=weight,
        training=False,
    )
    assert out.numpy()
    out = loss.discriminator_loss(
        transformer=transf,
        discriminator=disc,
        source=source,
        target=target,
        sample_weight=weight,
        training=False,
    )
    assert out.numpy()
