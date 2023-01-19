import tensorflow as tf
TF_FLOAT = tf.float32
from calotron.simulators.Simulator import Simulator


class ExportSimulator(tf.Module):
  def __init__(self, simulator):
    super().__init__()
    if not isinstance(simulator, Simulator):
      raise TypeError(f"`transformer` should be a calotron's "
                      f"`{type(Simulator).__name__}`, instead "
                      f"{type(simulator)} passed")
    self._simulator = simulator

  @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=TF_FLOAT)])
  def __call__(self, source, max_length):
    source = tf.cast(source, dtype=TF_FLOAT)
    result = self._simulator(source, max_length)
    return result

  @property
  def simulator(self) -> Simulator:
    return self._simulator
