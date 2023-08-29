import tensorflow as tf
from tensorflow import keras


class ConvDeepSets(keras.Model):
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
        assert (latent_dim % 2) == 0
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
                keras.layers.Conv1D(
                    filters=self._filters,
                    kernel_size=self._kernel_size,
                    strides=self._strides,
                    activation="relu",
                    kernel_initializer="glorot_uniform",
                    bias_initializer="zeros",
                    name=f"conv1D_{i}" if name else None,
                    dtype=self.dtype,
                )
            )
        self._seq.append(
            keras.layers.Dense(
                units=self._latent_dim,
                activation="relu",
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
                name=f"dense_{self._num_conv_layers}" if name else None,
                dtype=self.dtype,
            )
        )
        self._seq.append(
            keras.layers.Dropout(
                self._dropout_rate,
                name=f"dropout_{self._num_conv_layers}" if name else None,
                dtype=self.dtype,
            )
        )
        self._seq.append(
            keras.layers.Dense(
                int(self._latent_dim / 2),
                activation=None,
                kernel_initializer="he_normal",
                bias_initializer="zeros",
                name="dense_out" if name else None,
                dtype=self.dtype,
            )
        )

        # Final layers
        self._avg_pool = keras.layers.GlobalAveragePooling1D(name="avg_pool" if name else None)
        self._max_pool = keras.layers.GlobalMaxPooling1D(name="max_pool" if name else None)
        self._concat = keras.layers.Concatenate(name="concat" if name else None)

    def call(self, x, padding_mask=None) -> tf.Tensor:
        if padding_mask is not None:
            padding_mask = tf.tile(padding_mask[:, :, None], (1, 1, tf.shape(x)[2]))
            x *= padding_mask
        for layer in self._seq:
            x = layer(x)
        x_avg = self._avg_pool(x)
        x_max = self._max_pool(x)
        out = self._concat([x_avg, x_max])
        return out

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
