import tensorflow as tf

from calotron.simulators.Simulator import Simulator

TF_FLOAT = tf.float32


class ExportSimulator(tf.Module):
    def __init__(self, simulator, max_length, name=None):
        super().__init__(name=name)

        # Simulator
        if not isinstance(simulator, Simulator):
            raise TypeError(
                "`simulator` should be a calotron's `Simulator`, "
                f"instead {type(simulator)} passed"
            )
        self._simulator = simulator
        self._dtype = simulator._dtype

        # Sequence max length
        assert max_length >= 1
        self._max_length = int(max_length)

    @tf.function
    def __call__(self, dataset):
        assert isinstance(dataset, tf.data.Dataset)

        ta_target = tf.TensorArray(dtype=self._dtype, size=0, dynamic_size=True)

        idx = 0
        for source in dataset:
            source = tf.cast(source, dtype=self._dtype)
            ta_target = ta_target.write(
                index=idx, value=self._simulator(source, self._max_length)
            )
            idx += 1
        output = ta_target.stack()
        output = tf.reshape(
            output,
            (idx * tf.shape(output)[1], tf.shape(output)[2], tf.shape(output)[3]),
        )
        return output

    @property
    def simulator(self) -> Simulator:
        return self._simulator

    @property
    def max_length(self) -> int:
        return self._max_length
