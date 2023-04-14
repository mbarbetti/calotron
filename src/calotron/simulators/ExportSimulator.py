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
        ta_weight = tf.TensorArray(dtype=self._dtype, size=0, dynamic_size=True)

        idx = 0
        for source in dataset:
            source = tf.cast(source, dtype=self._dtype)
            out_target, attn_weights = self._simulator(source, self._max_length)

            ta_target = ta_target.write(index=idx, value=out_target)
            ta_weight = ta_weight.write(index=idx, value=attn_weights)
            idx += 1

        out_target = ta_target.stack()
        out_target = tf.reshape(
            out_target,
            shape=(
                idx * tf.shape(out_target)[1],
                tf.shape(out_target)[2],
                tf.shape(out_target)[3],
            ),
        )
        attn_weights = ta_weight.stack()
        attn_weights = tf.reshape(
            attn_weights,
            shape=(
                idx * tf.shape(attn_weights)[1],
                tf.shape(attn_weights)[2],
                tf.shape(attn_weights)[3],
                tf.shape(attn_weights)[4],
            ),
        )
        return out_target, attn_weights

    @property
    def simulator(self) -> Simulator:
        return self._simulator

    @property
    def max_length(self) -> int:
        return self._max_length
