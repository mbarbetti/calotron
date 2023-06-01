import tensorflow as tf


class MultilayerPerceptron(tf.keras.layers.Layer):
    def __init__(
        self, output_units, hidden_units, dropout_rate=0.1, name=None, dtype=None
    ) -> None:
        super().__init__(name=name, dtype=dtype)
        if name is not None:
            prefix = name.split("_")[0]
            suffix = name.split("_")[-1]

        # Output units
        assert isinstance(output_units, (int, float))
        assert output_units >= 1
        self._output_units = int(output_units)

        # Hidden units
        assert isinstance(hidden_units, (int, float))
        assert hidden_units >= 1
        self._hidden_units = int(hidden_units)

        # Dropout rate
        assert isinstance(dropout_rate, (int, float))
        assert dropout_rate >= 0.0 and dropout_rate < 1.0
        self._dropout_rate = float(dropout_rate)

        # Layer normalization
        self._layer_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)

        # Multilayer perceptron layers
        self._seq = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    units=self._hidden_units,
                    activation="relu",
                    kernel_initializer="glorot_normal",
                    bias_initializer="zeros",
                    name=f"{prefix}_mlp_dense_in_{suffix}" if name else None,
                    dtype=self.dtype,
                ),
                tf.keras.layers.Dense(
                    units=self._output_units,
                    activation="linear",
                    kernel_initializer="truncated_normal",
                    bias_initializer="zeros",
                    name=f"{prefix}_mlp_dense_out_{suffix}" if name else None,
                    dtype=self.dtype,
                ),
                tf.keras.layers.Dropout(
                    rate=self._dropout_rate,
                    name=f"{prefix}_mlp_dropout_{suffix}" if name else None,
                    dtype=self.dtype,
                ),
            ]
        )
        self._add = tf.keras.layers.Add()

    def call(self, x) -> tf.Tensor:
        norm_x = self._layer_norm(x)
        output = self._seq(norm_x)
        output = self._add([x, output])
        return output

    @property
    def output_units(self) -> int:
        return self._output_units

    @property
    def hidden_units(self) -> int:
        return self._hidden_units

    @property
    def dropout_rate(self) -> float:
        return self._dropout_rate
