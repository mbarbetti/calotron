import numpy as np
import pytest
import tensorflow as tf

FROM_MEAN_TO_SUM = 2
ALPHA_TO_INF = 1e4
CHUNK_SIZE = int(1e4)

source_true = np.random.uniform(1.0, 0.5, size=(CHUNK_SIZE, 4))
target_true = np.random.uniform(0.4, 0.5, size=(CHUNK_SIZE, 8))
target_pred = np.random.uniform(0.2, 0.5, size=(CHUNK_SIZE, 8))

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
    ]
)


@pytest.fixture
def loss():
    from calotron.losses import RefinedMeanSquaredError

    loss_ = RefinedMeanSquaredError(
        alpha=0.1,
        discriminator_from_logits=False,
        discriminator_label_smoothing=0.0,
    )
    return loss_


###########################################################################


def test_loss_configuration(loss):
    from calotron.losses import RefinedMeanSquaredError

    assert isinstance(loss, RefinedMeanSquaredError)
    assert isinstance(loss.name, str)


@pytest.mark.parametrize("discriminator_from_logits", [False, True])
def test_loss_use_no_weights(discriminator_from_logits):
    from calotron.losses import RefinedMeanSquaredError

    loss = RefinedMeanSquaredError(
        alpha=ALPHA_TO_INF,
        discriminator_from_logits=discriminator_from_logits,
        discriminator_label_smoothing=0.0,
    )
    if discriminator_from_logits:
        model.add(tf.keras.layers.Dense(1, activation="tanh"))
    else:
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
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
    assert out1.numpy() * FROM_MEAN_TO_SUM > out2.numpy() / ALPHA_TO_INF


@pytest.mark.parametrize("discriminator_from_logits", [False, True])
def test_loss_use_with_weights(discriminator_from_logits):
    w = np.random.uniform(0.0, 1.0, size=(CHUNK_SIZE, 1)) > 0.5
    from calotron.losses import RefinedMeanSquaredError

    loss = RefinedMeanSquaredError(
        alpha=ALPHA_TO_INF,
        discriminator_from_logits=discriminator_from_logits,
        discriminator_label_smoothing=0.0,
    )
    if discriminator_from_logits:
        model.add(tf.keras.layers.Dense(1, activation="tanh"))
    else:
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
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
    assert out1.numpy() * FROM_MEAN_TO_SUM > out2.numpy() / ALPHA_TO_INF
