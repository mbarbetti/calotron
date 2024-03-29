import numpy as np
import pytest
import tensorflow as tf

CHUNK_SIZE = int(1e4)

X = np.c_[
    np.random.uniform(-1, 1, size=CHUNK_SIZE),
    np.random.normal(0, 1, size=CHUNK_SIZE),
    np.random.exponential(5, size=CHUNK_SIZE),
]
Y = np.tanh(X[:, 0]) + 2 * X[:, 1] * X[:, 2]

model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(3,)))
for units in [16, 16, 16]:
    model.add(tf.keras.layers.Dense(units, activation="relu"))
model.add(tf.keras.layers.Dense(1))

adam = tf.keras.optimizers.Adam(learning_rate=0.001)
mse = tf.keras.losses.MeanSquaredError()


@pytest.fixture
def scheduler():
    from calotron.callbacks.schedulers import LearnRateCosineDecay

    sched = LearnRateCosineDecay(
        optimizer=adam,
        decay_steps=1000,
        alpha=0.95,
        min_learning_rate=0.001,
        verbose=True,
        key="lr",
    )
    return sched


###########################################################################


def test_sched_configuration(scheduler):
    from calotron.callbacks.schedulers import LearnRateCosineDecay

    assert isinstance(scheduler, LearnRateCosineDecay)
    assert isinstance(scheduler.name, str)
    assert isinstance(scheduler.optimizer, tf.keras.optimizers.Optimizer)
    assert isinstance(scheduler.decay_steps, int)
    assert isinstance(scheduler.alpha, float)
    assert isinstance(scheduler.min_learning_rate, float)
    assert isinstance(scheduler.verbose, bool)
    assert isinstance(scheduler.key, str)


@pytest.mark.parametrize("min_learning_rate", [None, 0.0005])
def test_sched_use(min_learning_rate):
    from calotron.callbacks.schedulers import LearnRateCosineDecay

    scheduler = LearnRateCosineDecay(
        optimizer=adam,
        decay_steps=1000,
        alpha=0.95,
        min_learning_rate=min_learning_rate,
        verbose=True,
    )
    model.compile(optimizer=adam, loss=mse)
    history = model.fit(X, Y, batch_size=500, epochs=5, callbacks=[scheduler])
    last_lr = float(f"{history.history['lr'][-1]:.4f}")
    if min_learning_rate is not None:
        assert last_lr == 0.0005
    else:
        assert last_lr == 0.0001
