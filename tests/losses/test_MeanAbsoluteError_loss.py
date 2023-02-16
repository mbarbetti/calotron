import numpy as np
import pytest
import tensorflow as tf

CHUNK_SIZE = int(1e4)

target_true = np.random.uniform(0.5, 1.0, size=(CHUNK_SIZE, 5))
target_pred = np.random.uniform(0.2, 0.8, size=(CHUNK_SIZE, 5))

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="relu"),
    ]
)


@pytest.fixture
def loss():
    from calotron.losses import MeanAbsoluteError

    loss_ = MeanAbsoluteError(reduction="auto", name="mae_loss")
    return loss_


###########################################################################


def test_loss_configuration(loss):
    from calotron.losses import MeanAbsoluteError

    assert isinstance(loss, MeanAbsoluteError)
    assert isinstance(loss.name, str)


def test_loss_use_no_weights(loss):
    out1 = loss.discriminator_loss(
        discriminator=model,
        target_true=target_true,
        target_pred=target_pred,
        sample_weight=None,
    )
    out2 = loss.transformer_loss(
        discriminator=model,
        target_true=target_true,
        target_pred=target_pred,
        sample_weight=None,
    )
    assert out1.numpy() == -out2.numpy()


def test_loss_use_with_weights(loss):
    w = np.random.uniform(0.0, 1.0, size=(CHUNK_SIZE, 1)) > 0.5
    out1 = loss.discriminator_loss(
        discriminator=model,
        target_true=target_true,
        target_pred=target_pred,
        sample_weight=w,
    )
    out2 = loss.transformer_loss(
        discriminator=model,
        target_true=target_true,
        target_pred=target_pred,
        sample_weight=w,
    )
    assert out1.numpy() == -out2.numpy()
