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
    from calotron.callbacks.schedulers import LearnRateAttnDecay

    sched = LearnRateAttnDecay(
        optimizer=adam, d_model=100, warmup_steps=4000, verbose=True, key="lr"
    )
    return sched


###########################################################################


def test_sched_configuration(scheduler):
    from calotron.callbacks.schedulers import LearnRateAttnDecay

    assert isinstance(scheduler, LearnRateAttnDecay)
    assert isinstance(scheduler.name, str)
    assert isinstance(scheduler.optimizer, tf.keras.optimizers.Optimizer)
    assert isinstance(scheduler.d_model, int)
    assert isinstance(scheduler.warmup_steps, int)
    assert isinstance(scheduler.verbose, bool)
    assert isinstance(scheduler.key, str)


def test_sched_use(scheduler):
    model.compile(optimizer=adam, loss=mse)
    history = model.fit(X, Y, batch_size=50, epochs=5, callbacks=[scheduler])
    last_lr = float(f"{history.history['lr'][-1]:.4f}")
    assert last_lr == 0.0004
