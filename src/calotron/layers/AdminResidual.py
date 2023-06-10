import math

import tensorflow as tf

OUTPUT_CHANGE_SCALE = ["O(n)", "O(logn)", "O(1)"]


class AdminResidual(tf.keras.layers.Layer):
    def __init__(
        self,
        embed_dim,
        num_res_layers,
        output_change_scale="O(logn)",
        name=None,
        dtype=None,
    ) -> None:
        super().__init__(name=name, dtype=dtype)

        # Embedding dimension
        assert isinstance(embed_dim, (int, float))
        assert embed_dim >= 1
        self._embed_dim = int(embed_dim)

        # Number of residual layers
        assert isinstance(num_res_layers, (int, float))
        assert num_res_layers >= 1
        self._num_res_layers = int(num_res_layers)

        # Output change scale
        assert isinstance(output_change_scale, str)
        if output_change_scale not in OUTPUT_CHANGE_SCALE:
            raise ValueError(
                "`output_change_scale` should be selected "
                f"in {OUTPUT_CHANGE_SCALE}, instead "
                f"'{output_change_scale}' passed"
            )
        self._output_change_scale = output_change_scale

        self._omega = self._compute_init_value()
        self._add = tf.keras.layers.Add()

    def _compute_init_value(self) -> tf.Variable:
        if self._output_change_scale == "O(n)":
            omega_value = 1.0
            trainable = False
        elif self._output_change_scale == "O(logn)":
            omega_value = (self._num_res_layers + 1.0) / math.log(
                self._num_res_layers + 1.0
            ) - 1.0
            trainable = True
        elif self._output_change_scale == "O(1)":
            omega_value = self._num_res_layers
            trainable = True
        omega = tf.ones(shape=(self._embed_dim)) * omega_value**0.5
        return tf.Variable(omega, trainable=trainable, dtype=self.dtype)

    def call(self, x, f_x) -> tf.Tensor:
        x *= tf.tile(self._omega[None, None, :], (tf.shape(x)[0], tf.shape(x)[1], 1))
        output = self._add([x, f_x])
        return output

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def num_res_layers(self) -> int:
        return self._num_res_layers

    @property
    def output_change_scale(self) -> str:
        return self._output_change_scale
