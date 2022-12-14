import pytest
import numpy as np
import tensorflow as tf


chunk_size = int(1e5)

X = np.c_[
  np.random.uniform(-1, 1, size=chunk_size),
  np.random.normal(0, 1, size=chunk_size),
  np.random.exponential(5, size=chunk_size)
]
Y = np.tanh(X[:,0]) + 2 * X[:,1] * X[:,2]

model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(3,)))
for units in [16, 16, 16]:
  model.add(tf.keras.layers.Dense(units, activation="relu"))
model.add(tf.keras.layers.Dense(1))


@pytest.fixture
def scheduler(cycle=False):
  from calotron.callbacks.schedulers import PolynomialDecay
  sched = PolynomialDecay(
              decay_steps=1000,
              end_learning_rate=0.0001,
              power=1.0,
              cycle=cycle)
  return sched


###########################################################################


def test_sched_configuration(scheduler):
  from calotron.callbacks.schedulers import PolynomialDecay
  assert isinstance(scheduler, PolynomialDecay)
  assert isinstance(scheduler.decay_steps, int)
  assert isinstance(scheduler.end_learning_rate, float)
  assert isinstance(scheduler.power, float)
  assert isinstance(scheduler.cycle, bool)


@pytest.mark.parametrize("cycle", [False, True])
def test_sched_use(cycle):
  from calotron.callbacks.schedulers import PolynomialDecay
  sched = PolynomialDecay(
              decay_steps=1000,
              end_learning_rate=0.0001,
              power=1.0,
              cycle=cycle)
  adam = tf.keras.optimizers.Adam(learning_rate=0.001)
  mse = tf.keras.losses.MeanSquaredError()
  model.compile(optimizer=adam, loss=mse)
  history = model.fit(X, Y, batch_size=512, epochs=10, callbacks=[sched])
  assert isinstance(sched._dtype, np.dtype)
  last_lr = float(history.history["lr"][-1])
  assert  last_lr < 0.001
