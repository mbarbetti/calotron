import numpy as np
import pytest
import tensorflow as tf

CHUNK_SIZE = int(1e4)

source_true = np.random.uniform(1.0, 0.5, size=(CHUNK_SIZE, 8, 5))
target_true = np.random.uniform(0.4, 0.5, size=(CHUNK_SIZE, 4, 3))
target_pred = np.random.uniform(0.2, 0.5, size=(CHUNK_SIZE, 4, 3))

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)


@pytest.fixture
def loss():
    from calotron.losses import KLDivergence

    loss_ = KLDivergence()
    return loss_


###########################################################################


def test_loss_configuration(loss):
    from calotron.losses import KLDivergence

    assert isinstance(loss, KLDivergence)
    assert isinstance(loss.name, str)


def test_loss_use_no_weights(loss):
    out1 = loss.discriminator_loss(
        discriminator=model,
        source_true=source_true,
        target_true=target_true,
        target_pred=target_pred,
        sample_weight=None,
    )
    out2 = loss.transformer_loss(
        discriminator=model,
        source_true=source_true,
        target_true=target_true,
        target_pred=target_pred,
        sample_weight=None,
    )
    assert out1.numpy() == -out2.numpy()


def test_loss_use_with_weights(loss):
    w = np.random.uniform(0.0, 1.0, size=(CHUNK_SIZE, 1)) > 0.5
    out1 = loss.discriminator_loss(
        discriminator=model,
        source_true=source_true,
        target_true=target_true,
        target_pred=target_pred,
        sample_weight=w,
    )
    out2 = loss.transformer_loss(
        discriminator=model,
        source_true=source_true,
        target_true=target_true,
        target_pred=target_pred,
        sample_weight=w,
    )
    assert out1.numpy() == -out2.numpy()
