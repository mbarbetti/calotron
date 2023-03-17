import tensorflow as tf


class FeedForward(tf.keras.layers.Layer):
    def __init__(
        self,
        output_units,
        hidden_units,
        dropout_rate=0.1,
        residual_smoothing=True,
        name=None,
        dtype=None,
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

        # Residual smoothing
        assert isinstance(residual_smoothing, bool)
        self._residual_smoothing = residual_smoothing

        # Smoothing layer
        if self._residual_smoothing:
            self._emb_layer = tf.keras.layers.Dense(
                units=self._output_units,
                name=f"{prefix}_fnn_res_smoothing_{suffix}" if name else None,
                dtype=self.dtype,
            )
        else:
            self._emb_layer = None

        # Feed-forward net layers
        self._seq = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    units=self._hidden_units,
                    activation="relu",
                    kernel_initializer="glorot_uniform",
                    name=f"{prefix}_fnn_dense_in_{suffix}" if name else None,
                    dtype=self.dtype,
                ),
                tf.keras.layers.Dense(
                    units=self._output_units,
                    activation="linear",
                    kernel_initializer="he_normal",
                    name=f"{prefix}_fnn_dense_out_{suffix}" if name else None,
                    dtype=self.dtype,
                ),
                tf.keras.layers.Dropout(
                    rate=self._dropout_rate,
                    name=f"{prefix}_fnn_dropout_{suffix}" if name else None,
                    dtype=self.dtype,
                ),
            ]
        )
        self._add = tf.keras.layers.Add()
        # self._layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x) -> tf.Tensor:
        if self._emb_layer is not None:
            x = self._emb_layer(x)
        fnn_output = self._seq(x)
        output = self._add([x, fnn_output])
        # output = self._layer_norm(output)
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

    @property
    def residual_smoothing(self) -> bool:
        return self._residual_smoothing
