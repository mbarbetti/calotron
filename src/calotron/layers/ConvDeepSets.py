import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Layer


class ConvDeepSets(Layer):
    def __init__(
        self,
        latent_dim,
        num_conv_layers,
        filters=64,
        kernel_size=4,
        strides=4,
        dropout_rate=0.0,
        name=None,
        dtype=None,
    ) -> None:
        super().__init__(name=name, dtype=dtype)

        # Latent space dimension
        assert isinstance(latent_dim, (int, float))
        assert latent_dim >= 1
        self._latent_dim = int(latent_dim)

        # Number of convolutional layers
        assert isinstance(num_conv_layers, (int, float))
        assert num_conv_layers >= 1
        self._num_conv_layers = int(num_conv_layers)

        # Convolutional filters
        assert isinstance(filters, (int, float))
        assert filters >= 1
        self._filters = int(filters)

        # Convolutional kernel size
        assert isinstance(kernel_size, (int, float))
        assert kernel_size >= 1
        self._kernel_size = int(kernel_size)

        # Convolutional strides
        assert isinstance(strides, (int, float))
        assert strides >= 1
        self._strides = int(strides)

        # Dropout rate
        assert isinstance(dropout_rate, (int, float))
        assert dropout_rate >= 0.0 and dropout_rate < 1.0
        self._dropout_rate = float(dropout_rate)

        # Convolutional Deep Sets layers
        self._seq = list()
        for i in range(self._num_conv_layers):
            self._seq.append(
                Conv1D(
                    filters=self._filters,
                    kernel_size=self._kernel_size,
                    strides=self._strides,
                    activation="relu",
                    kernel_initializer="glorot_uniform",
                    bias_initializer="zeros",
                    name=f"ds_conv1D_{i}",
                    dtype=self.dtype,
                )
            )
        self._seq.append(
            Dense(
                units=2 * self._latent_dim,
                activation="relu",
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
                name=f"ds_dense_{self._num_conv_layers}",
                dtype=self.dtype,
            )
        )
        self._seq.append(
            Dropout(
                self._dropout_rate,
                name=f"ds_dropout_{self._num_conv_layers}",
                dtype=self.dtype,
            )
        )
        self._seq.append(
            Dense(
                self._latent_dim,
                activation=None,
                kernel_initializer="truncated_normal",
                bias_initializer="zeros",
                name="ds_dense_out",
                dtype=self.dtype,
            )
        )

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
    def num_conv_layers(self) -> int:
        return self._num_conv_layers

    @property
    def filters(self) -> int:
        return self._filters

    @property
    def kernel_size(self) -> int:
        return self._kernel_size

    @property
    def strides(self) -> int:
        return self._strides

    @property
    def dropout_rate(self) -> float:
        return self._dropout_rate
