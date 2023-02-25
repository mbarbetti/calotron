import numpy as np
import tensorflow as tf

from calotron.layers import Decoder, Encoder, MultiActivations

START_TOKEN_INITIALIZERS = ["zeros", "ones", "means"]


class Transformer(tf.keras.Model):
    def __init__(
        self,
        output_depth,
        encoder_depth,
        decoder_depth,
        num_layers,
        num_heads,
        key_dims=None,
        pos_dims=None,
        pos_norms=128,
        max_lengths=32,
        ff_units=256,
        dropout_rates=0.1,
        pos_sensitive=False,
        residual_smoothing=True,
        output_activations=None,
        start_token_initializer="zeros",
        name=None,
        dtype=None,
    ) -> None:
        super().__init__(name=name, dtype=dtype)

        # Output depth
        assert output_depth >= 1
        self._output_depth = int(output_depth)

        # Encoder depth
        assert encoder_depth >= 1
        self._encoder_depth = int(encoder_depth)

        # Decoder depth
        assert decoder_depth >= 1
        self._decoder_depth = int(decoder_depth)

        # Number of layers (encoder/decoder)
        if isinstance(num_layers, (int, float)):
            assert num_layers >= 1
            self._num_layers = [int(num_layers)] * 2
        else:
            assert isinstance(num_layers, (list, tuple, np.ndarray))
            assert len(num_layers) == 2
            self._num_layers = list()
            for num in num_layers:
                assert isinstance(num, (int, float))
                assert num >= 1
                self._num_layers.append(int(num))

        # Number of heads (encoder/decoder)
        if isinstance(num_heads, (int, float)):
            assert num_heads >= 1
            self._num_heads = [int(num_heads)] * 2
        else:
            assert isinstance(num_heads, (list, tuple, np.ndarray))
            assert len(num_heads) == 2
            self._num_heads = list()
            for num in num_heads:
                assert isinstance(num, (int, float))
                assert num >= 1
                self._num_heads.append(int(num))

        # Key dimension (encoder/decoder)
        if key_dims:
            if isinstance(key_dims, (int, float)):
                assert key_dims >= 1
                self._key_dims = [int(key_dims)] * 2
            else:
                assert isinstance(key_dims, (list, tuple, np.ndarray))
                assert len(key_dims) == 2
                self._key_dims = list()
                for dim in key_dims:
                    if dim:
                        assert isinstance(dim, (int, float))
                        assert dim >= 1
                        dim = int(dim)
                    self._key_dims.append(dim)
        else:
            self._key_dims = [None, None]

        # Position dimension (encoder/decoder)
        if pos_dims:
            if isinstance(pos_dims, (int, float)):
                assert pos_dims >= 1
                self._pos_dims = [int(pos_dims)] * 2
            else:
                assert isinstance(pos_dims, (list, tuple, np.ndarray))
                assert len(pos_dims) == 2
                self._pos_dims = list()
                for dim in pos_dims:
                    if dim:
                        assert isinstance(dim, (int, float))
                        assert dim >= 1
                        dim = int(dim)
                    self._pos_dims.append(dim)
        else:
            self._pos_dims = [None, None]

        # Position normalization (encoder/decoder)
        if isinstance(pos_norms, (int, float)):
            assert pos_norms > 0.0
            self._pos_norms = [float(pos_norms)] * 2
        else:
            assert isinstance(pos_norms, (list, tuple, np.ndarray))
            assert len(pos_norms) == 2
            self._pos_norms = list()
            for norm in pos_norms:
                assert isinstance(norm, (int, float))
                assert norm > 0.0
                self._pos_norms.append(float(norm))

        # Max length (encoder/decoder)
        if isinstance(max_lengths, (int, float)):
            assert max_lengths >= 1
            self._max_lengths = [int(max_lengths)] * 2
        else:
            assert isinstance(max_lengths, (list, tuple, np.ndarray))
            assert len(max_lengths) == 2
            self._max_lengths = list()
            for length in max_lengths:
                assert isinstance(length, (int, float))
                assert length >= 1
                self._max_lengths.append(int(length))

        # Feed-forward net units (encoder/decoder)
        if isinstance(ff_units, (int, float)):
            assert ff_units >= 1
            self._ff_units = [int(ff_units)] * 2
        else:
            assert isinstance(ff_units, (list, tuple, np.ndarray))
            assert len(ff_units) == 2
            self._ff_units = list()
            for units in ff_units:
                assert isinstance(units, (int, float))
                assert units >= 1
                self._ff_units.append(int(units))

        # Dropout rate (encoder/decoder)
        if isinstance(dropout_rates, (int, float)):
            assert dropout_rates >= 0.0 and dropout_rates < 1.0
            self._dropout_rates = [float(dropout_rates)] * 2
        else:
            assert isinstance(dropout_rates, (list, tuple, np.ndarray))
            assert len(dropout_rates) == 2
            self._dropout_rates = list()
            for rate in dropout_rates:
                assert isinstance(rate, (int, float))
                assert rate >= 0.0 and rate < 1.0
                self._dropout_rates.append(float(rate))

        # Position sensitive (encoder/decoder)
        if isinstance(pos_sensitive, bool):
            self._pos_sensitive = [pos_sensitive] * 2
        else:
            assert isinstance(pos_sensitive, (list, tuple, np.ndarray))
            assert len(pos_sensitive) == 2
            self._pos_sensitive = list()
            for flag in pos_sensitive:
                assert isinstance(flag, bool)
                self._pos_sensitive.append(flag)

        # Residual blocks smoothing (encoder/decoder)
        if isinstance(residual_smoothing, bool):
            assert isinstance(residual_smoothing, bool)
            self._residual_smoothing = [residual_smoothing] * 2
        else:
            assert isinstance(residual_smoothing, (list, tuple, np.ndarray))
            assert len(residual_smoothing) == 2
            self._residual_smoothing = list()
            for flag in residual_smoothing:
                assert isinstance(flag, bool)
                self._residual_smoothing.append(flag)

        # Start token initializer
        if start_token_initializer not in START_TOKEN_INITIALIZERS:
            raise ValueError(
                "`start_token_initializer` should be selected "
                f"in {START_TOKEN_INITIALIZERS}, instead "
                f"'{start_token_initializer}' passed"
            )
        self._start_token_initializer = start_token_initializer

        # Encoder
        self._encoder = Encoder(
            encoder_depth=self._encoder_depth,
            num_layers=self._num_layers[0],
            num_heads=self._num_heads[0],
            key_dim=self._key_dims[0],
            pos_dim=self._pos_dims[0],
            pos_norm=self._pos_norms[0],
            max_length=self._max_lengths[0],
            ff_units=self._ff_units[0],
            dropout_rate=self._dropout_rates[0],
            pos_sensitive=self._pos_sensitive[0],
            residual_smoothing=self._residual_smoothing[0],
            dtype=self.dtype,
        )

        # Decoder
        self._decoder = Decoder(
            decoder_depth=self._decoder_depth,
            num_layers=self._num_layers[1],
            num_heads=self._num_heads[1],
            key_dim=self._key_dims[1],
            pos_dim=self._pos_dims[1],
            pos_norm=self._pos_norms[1],
            max_length=self._max_lengths[1],
            ff_units=self._ff_units[1],
            dropout_rate=self._dropout_rates[1],
            pos_sensitive=self._pos_sensitive[1],
            residual_smoothing=self._residual_smoothing[1],
            dtype=self.dtype,
        )

        # Final layers
        self._final_layer = tf.keras.layers.Dense(
            self._output_depth, name="output_layer", dtype=self.dtype
        )
        if output_activations is not None:
            self._multi_act_layer = MultiActivations(
                output_activations,
                self._output_depth,
                name="ma_layer",
                dtype=self.dtype,
            )
            self._output_activations = self._multi_act_layer.output_activations
        else:
            self._output_activations = None

    def call(self, inputs) -> tf.Tensor:
        source, target = inputs
        target = self._prepare_input_target(target)
        context = self._encoder(
            x=source
        )  # (batch_size, source_elements, encoder_depth)
        output = self._decoder(
            x=target, context=context
        )  # (batch_size, target_elements, decoder_depth)
        output = self._final_layer(
            output
        )  # (batch_size, target_elements, output_depth)
        if self._output_activations is not None:
            output = self._multi_act_layer(
                output
            )  # (batch_size, target_elements, output_depth)
        return output

    def _prepare_input_target(self, target) -> tf.Tensor:
        if self._start_token_initializer == "zeros":
            start_token = tf.zeros((tf.shape(target)[0], 1, tf.shape(target)[2]))
        elif self._start_token_initializer == "ones":
            start_token = tf.ones((tf.shape(target)[0], 1, tf.shape(target)[2]))
        elif self._start_token_initializer == "means":
            start_token = tf.reduce_mean(target, axis=(0, 1))[None, None, :]
            start_token = tf.tile(start_token, (tf.shape(target)[0], 1, 1))
        return tf.concat([start_token, target[:, :-1, :]], axis=1)

    def get_start_token(self, target) -> tf.Tensor:
        if self._start_token_initializer == "zeros":
            start_token = tf.zeros((tf.shape(target)[0], tf.shape(target)[2]))
        elif self._start_token_initializer == "ones":
            start_token = tf.ones((tf.shape(target)[0], tf.shape(target)[2]))
        elif self._start_token_initializer == "means":
            start_token = tf.reduce_mean(target, axis=(0, 1))[None, :]
            start_token = tf.tile(start_token, (tf.shape(target)[0], 1))
        return start_token

    @property
    def output_depth(self) -> int:
        return self._output_depth

    @property
    def encoder_depth(self) -> int:
        return self._encoder_depth

    @property
    def decoder_depth(self) -> int:
        return self._decoder_depth

    @property
    def num_layers(self) -> list:
        return self._num_layers

    @property
    def num_heads(self) -> list:
        return self._num_heads

    @property
    def key_dims(self) -> list:
        return self._key_dims

    @property
    def pos_dims(self) -> list:
        return self._pos_dims

    @property
    def pos_norms(self) -> list:
        return self._pos_norms

    @property
    def max_lengths(self) -> list:
        return self._max_lengths

    @property
    def ff_units(self) -> list:
        return self._ff_units

    @property
    def dropout_rates(self) -> list:
        return self._dropout_rates

    @property
    def pos_sensitive(self) -> list:
        return self._pos_sensitive

    @property
    def residual_smoothing(self) -> list:
        return self._residual_smoothing

    @property
    def start_token_initializer(self) -> str:
        return self._start_token_initializer

    @property
    def output_activations(self):  # TODO: add Union[list, None]
        return self._output_activations
