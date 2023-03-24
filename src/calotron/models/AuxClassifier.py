import tensorflow as tf

from calotron.models.Transformer import Transformer


class AuxClassifier(tf.keras.Model):
    def __init__(
        self,
        transformer,
        output_depth,
        output_activation=None,
        dropout_rate=0.1,
        name=None,
        dtype=None,
    ) -> None:
        super().__init__(name=name, dtype=dtype)

        # Transformer's Encoder
        if not isinstance(transformer, Transformer):
            raise TypeError(
                f"`transformer` should be a calotron's `Transformer`, "
                f"instead {type(transformer)} passed"
            )
        self._encoder = transformer._encoder

        # Output depth
        assert output_depth >= 1
        self._output_depth = int(output_depth)

        # Output activation
        self._output_activation = output_activation

        # Dropout rate
        assert isinstance(dropout_rate, (int, float))
        assert dropout_rate >= 0.0 and dropout_rate < 1.0
        self._dropout_rate = float(dropout_rate)

        # Output layers
        self._seq = [
            tf.keras.layers.Dropout(
                self._dropout_rate, name="a_dropout", dtype=self.dtype
            ),
            tf.keras.layers.Dense(
                self._output_depth,
                activation=self._output_activation,
                kernel_initializer="glorot_uniform",
                name="a_output_layer",
                dtype=self.dtype,
            ),
        ]

    def call(self, x) -> tf.Tensor:
        output = self._encoder(x)
        for layer in self._seq:
            output = layer(output)
        return output

    @property
    def output_depth(self) -> int:
        return self._output_depth

    @property
    def output_activation(self):  # TODO: add Union[None, activation]
        return self.output_activation

    @property
    def dropout_rate(self) -> float:
        return self._dropout_rate
