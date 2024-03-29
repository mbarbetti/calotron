import tensorflow as tf
from tensorflow import keras

from calotron.utils.checks import checkActivations


class MultiActivations(keras.layers.Layer):
    def __init__(self, activations, output_depth, name=None, dtype=None) -> None:
        super().__init__(name=name, dtype=dtype)

        # Output depth
        assert isinstance(output_depth, (int, float))
        assert output_depth >= 1
        self._output_depth = int(output_depth)

        # Output activations
        self._output_activations = checkActivations(activations, output_depth, dtype)

        self._concat = keras.layers.Concatenate(name="ma_concat" if name else None)

    def call(self, x) -> tf.Tensor:
        if x.shape[2] != self._output_depth:
            raise ValueError(
                f"`output_depth` passed {self._output_depth} doesn't "
                f"match with the input tensor shape ({x.shape})"
            )
        if self._output_activations is not None:
            concat = list()
            for i, activation in enumerate(self._output_activations):
                concat.append(activation(x[:, :, i])[:, :, None])
            return self._concat(concat)
        else:
            return x

    @property
    def output_activations(self):  # TODO: add Union[list, None]
        return self._output_activations

    @property
    def output_depth(self) -> int:
        return self._output_depth
