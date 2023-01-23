import numpy as np
import pytest
import tensorflow as tf

np.random.seed(42)
chunk_size = int(1e4)
target_true = np.random.uniform(0.5, 1.0, size=(chunk_size, 5))
target_pred = np.random.uniform(0.2, 0.8, size=(chunk_size, 5))

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
    ]
)


@pytest.fixture
def loss():
    from calotron.losses import BinaryCrossentropy

    loss_ = BinaryCrossentropy(
        from_logits=False,
        label_smoothing=0.0,
        axis=-1,
        reduction="auto",
        name="bce_loss",
    )
    return loss_


###########################################################################


def test_loss_configuration(loss):
    from calotron.losses import BinaryCrossentropy

    assert isinstance(loss, BinaryCrossentropy)
    assert isinstance(loss.name, str)


@pytest.mark.parametrize("from_logits", [False, True])
def test_loss_use_no_weights(from_logits):
    from calotron.losses import BinaryCrossentropy

    loss = BinaryCrossentropy(
        from_logits=from_logits,
        label_smoothing=0.0,
        axis=-1,
        reduction="auto",
        name="bce_loss",
    )
    if from_logits:
        model.add(tf.keras.layers.Dense(1, activation="tanh"))
    else:
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
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
    assert out1.numpy() > out2.numpy()


@pytest.mark.parametrize("from_logits", [False, True])
def test_loss_use_with_weights(from_logits):
    w = np.random.uniform(0.0, 1.0, size=(chunk_size, 1)) > 0.5
    from calotron.losses import BinaryCrossentropy

    loss = BinaryCrossentropy(
        from_logits=from_logits,
        label_smoothing=0.0,
        axis=-1,
        reduction="auto",
        name="bce_loss",
    )
    if from_logits:
        model.add(tf.keras.layers.Dense(1, activation="tanh"))
    else:
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
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
    assert out1.numpy() > out2.numpy()
