import tensorflow as tf

from calotron.simulators.Simulator import Simulator

TF_FLOAT = tf.float32


class ExportSimulator(tf.Module):
    def __init__(self, simulator, max_length, name=None):
        super().__init__(name=name)
        if not isinstance(simulator, Simulator):
            raise TypeError(
                "`simulator` should be a calotron's `Simulator`, "
                f"instead {type(simulator)} passed"
            )
        self._simulator = simulator
        assert max_length >= 1
        self._max_length = int(max_length)

    @tf.function
    def __call__(self, source):
        source = tf.cast(source, dtype=TF_FLOAT)
        result = self._simulator(source, self._max_length)
        return result

    @property
    def simulator(self) -> Simulator:
        return self._simulator
    
    @property
    def max_length(self) -> int:
        return self._max_length
