import numpy as np
import pytest
import tensorflow as tf

from calotron.models import Discriminator, Transformer

FROM_MEAN_TO_SUM = 2
ALPHA_TO_INF = 1e4
CHUNK_SIZE = int(1e4)


source = tf.random.normal(shape=(CHUNK_SIZE, 8, 5))
target = tf.random.normal(shape=(CHUNK_SIZE, 4, 3))


transf = Transformer(
    output_depth=target.shape[2],
    encoder_depth=8,
    decoder_depth=8,
    num_layers=2,
    num_heads=4,
    key_dims=32,
    fnn_units=16,
    dropout_rates=0.1,
    seq_ord_latent_dims=16,
    seq_ord_max_lengths=[source.shape[1], target.shape[1]],
    seq_ord_normalizations=10_000,
    residual_smoothing=True,
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
    from calotron.losses import GlobalEventReco

    loss_ = GlobalEventReco(
        alpha=0.1, noise_stddev=0.05, from_logits=False, label_smoothing=0.0
    )
    return loss_


###########################################################################


def test_loss_configuration(loss):
    from calotron.losses import GlobalEventReco

    assert isinstance(loss, GlobalEventReco)
    assert isinstance(loss.alpha, float)
    assert isinstance(loss.noise_stddev, float)
    assert isinstance(loss.from_logits, bool)
    assert isinstance(loss.label_smoothing, float)
    assert isinstance(loss.name, str)


@pytest.mark.parametrize("from_logits", [False, True])
def test_loss_use_no_weights(from_logits):
    from calotron.losses import GlobalEventReco

    loss = GlobalEventReco(
        alpha=ALPHA_TO_INF,
        noise_stddev=0.05,
        from_logits=from_logits,
        label_smoothing=0.0,
    )
    if from_logits:
        disc._seq += [tf.keras.layers.Dense(1, activation="tanh")]
    else:
        disc._seq += [tf.keras.layers.Dense(1, activation="sigmoid")]
    out1 = loss.discriminator_loss(
        transformer=transf,
        discriminator=disc,
        source=source,
        target=target,
        sample_weight=None,
        training=False,
    )
    out2 = loss.transformer_loss(
        transformer=transf,
        discriminator=disc,
        source=source,
        target=target,
        sample_weight=None,
        training=False,
    )
    assert out1.numpy() * FROM_MEAN_TO_SUM > out2.numpy() / ALPHA_TO_INF


@pytest.mark.parametrize("from_logits", [False, True])
def test_loss_use_with_weights(from_logits):
    w = np.random.uniform(0.0, 1.0, size=(CHUNK_SIZE, 1))
    from calotron.losses import GlobalEventReco

    loss = GlobalEventReco(
        alpha=ALPHA_TO_INF,
        noise_stddev=0.05,
        from_logits=from_logits,
        label_smoothing=0.0,
    )
    if from_logits:
        disc._seq += [tf.keras.layers.Dense(1, activation="tanh")]
    else:
        disc._seq += [tf.keras.layers.Dense(1, activation="sigmoid")]
    out1 = loss.discriminator_loss(
        transformer=transf,
        discriminator=disc,
        source=source,
        target=target,
        sample_weight=w,
        training=False,
    )
    out2 = loss.transformer_loss(
        transformer=transf,
        discriminator=disc,
        source=source,
        target=target,
        sample_weight=w,
        training=False,
    )
    assert out1.numpy() * FROM_MEAN_TO_SUM > out2.numpy() / ALPHA_TO_INF
