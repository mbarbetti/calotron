import tensorflow as tf
from tensorflow import keras

from calotron.layers.AdminResidual import AdminResidual

LN_EPSILON = 0.001


class MultilayerPerceptron(keras.layers.Layer):
    def __init__(
        self,
        output_units,
        hidden_units,
        num_res_layers,
        admin_res_scale="O(n)",
        dropout_rate=0.0,
        name=None,
        dtype=None,
    ) -> None:
        super().__init__(name=name, dtype=dtype)
        if name is not None:
            prefix = "_".join(w for w in name.split("_")[:-1])
            suffix = name.split("_")[-1]
        else:
            prefix, suffix = None, None

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

        # Multilayer perceptron layers
        self._seq = keras.Sequential(
            [
                keras.layers.Dense(
                    units=self._hidden_units,
                    activation="relu",
                    kernel_initializer="he_normal",
                    bias_initializer="zeros",
                    name=f"{prefix}_dense_in_{suffix}" if name else None,
                    dtype=self.dtype,
                ),
                keras.layers.Dense(
                    units=self._output_units,
                    activation=None,
                    kernel_initializer="he_normal",
                    bias_initializer="zeros",
                    name=f"{prefix}_dense_out_{suffix}" if name else None,
                    dtype=self.dtype,
                ),
                keras.layers.Dropout(
                    rate=self._dropout_rate,
                    name=f"{prefix}_dropout_{suffix}" if name else None,
                    dtype=self.dtype,
                ),
            ],
            name=f"{prefix}_seq_{suffix}" if name else None,
        )
        self._res = AdminResidual(
            embed_dim=output_units,
            num_res_layers=num_res_layers,
            output_change_scale=admin_res_scale,
            name=f"{prefix}_res_{suffix}" if name else None,
            dtype=self.dtype,
        )
        self._ln = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=LN_EPSILON,
            name=f"{prefix}_ln_{suffix}" if name else None,
            dtype=self.dtype,
        )

    def call(self, x) -> tf.Tensor:
        f_x = self._seq(x)
        res = self._res([x, f_x])
        out = self._ln(res)
        return out

    @property
    def output_units(self) -> int:
        return self._output_units

    @property
    def hidden_units(self) -> int:
        return self._hidden_units

    @property
    def num_res_layers(self) -> int:
        return self._res.num_res_layers

    @property
    def admin_res_scale(self) -> str:
        return self._res.output_change_scale

    @property
    def dropout_rate(self) -> float:
        return self._dropout_rate
