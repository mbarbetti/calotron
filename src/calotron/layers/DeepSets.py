import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Layer


class DeepSets(Layer):
    def __init__(
        self,
        latent_dim,
        dense_num_layers=0,
        dense_units=128,
        conv1D_num_layers=0,
        conv1D_filters=64,
        conv1D_kernel_size=4,
        conv1D_strides=4,
        dropout_rate=0.0,
        name=None,
        dtype=None,
    ) -> None:
        super().__init__(name=name, dtype=dtype)

        # Latent space dimension
        assert isinstance(latent_dim, (int, float))
        assert latent_dim >= 1
        self._latent_dim = int(latent_dim)

        # Number of Dense layers
        assert isinstance(dense_num_layers, (int, float))
        assert dense_num_layers >= 0
        self._dense_num_layers = int(dense_num_layers)

        # Number of Dense units
        assert isinstance(dense_units, (int, float))
        assert dense_units >= 1
        self._dense_units = int(dense_units)

        # Number of Conv1D layers
        assert isinstance(conv1D_num_layers, (int, float))
        assert conv1D_num_layers >= 0
        self._conv1D_num_layers = int(conv1D_num_layers)

        # Number of Conv1D filters
        assert isinstance(conv1D_filters, (int, float))
        assert conv1D_filters >= 1
        self._conv1D_filters = int(conv1D_filters)

        # Length of Conv1D kernel size
        assert isinstance(conv1D_kernel_size, (int, float))
        assert conv1D_kernel_size >= 1
        self._conv1D_kernel_size = int(conv1D_kernel_size)

        # Length of Conv1D strides
        assert isinstance(conv1D_strides, (int, float))
        assert conv1D_strides >= 1
        self._conv1D_strides = int(conv1D_strides)

        # Dropout rate
        assert isinstance(dropout_rate, (int, float))
        assert dropout_rate >= 0.0 and dropout_rate < 1.0
        self._dropout_rate = float(dropout_rate)

        # Hidden layers
        self._seq = list()
        for i in range(self._conv1D_num_layers):
            self._seq.append(
                Conv1D(
                    filters=self._conv1D_filters,
                    kernel_size=self._conv1D_kernel_size,
                    strides=self._conv1D_strides,
                    activation="relu",
                    kernel_initializer="glorot_uniform",
                    bias_initializer="zeros",
                    name=f"ds_conv1D_{i}",
                    dtype=self.dtype,
                )
            )

        for i in range(self._dense_num_layers):
            self._seq.append(
                Dense(
                    units=self._dense_units,
                    activation="relu",
                    kernel_initializer="glorot_uniform",
                    bias_initializer="zeros",
                    name=f"ds_dense_{i}",
                    dtype=self.dtype,
                )
            )
            self._seq.append(
                Dropout(self._dropout_rate, name=f"ds_dropout_{i}", dtype=self.dtype)
            )

        # Output layer
        self._seq += [
            Dense(
                self._latent_dim,
                activation="relu",
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
                name="ds_dense_out",
                dtype=self.dtype,
            )
        ]

    def call(self, x, padding_mask=None) -> tf.Tensor:
        if padding_mask is not None:
            padding_mask = tf.tile(padding_mask[:, :, None], (1, 1, tf.shape(x)[2]))
            x *= padding_mask
        for layer in self._seq:
            x = layer(x)
        output = tf.reduce_sum(x, axis=1)
        return output

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def dense_num_layers(self) -> int:
        return self._dense_num_layers

    @property
    def dense_units(self) -> int:
        return self._dense_units

    @property
    def conv1D_num_layers(self) -> int:
        return self._conv1D_num_layers

    @property
    def conv1D_filters(self) -> int:
        return self._conv1D_filters

    @property
    def conv1D_kernel_size(self) -> int:
        return self._conv1D_kernel_size

    @property
    def conv1D_strides(self) -> int:
        return self._conv1D_strides

    @property
    def dropout_rate(self) -> float:
        return self._dropout_rate
