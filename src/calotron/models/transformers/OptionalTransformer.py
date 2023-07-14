from calotron.layers import Decoder, Encoder
from calotron.models.transformers.Transformer import (
    START_TOKEN_INITIALIZERS,
    Transformer,
)


class OptionalTransformer(Transformer):
    def __init__(
        self,
        output_depth,
        encoder_options={
            "output_depth": 32,
            "num_layers": 5,
            "num_heads": 4,
            "key_dim": 64,
        },
        decoder_options={
            "output_depth": 32,
            "num_layers": 5,
            "num_heads": 4,
            "key_dim": 64,
        },
        output_activations=None,
        start_token_initializer="ones",
        name=None,
        dtype=None,
    ) -> None:
        super(Transformer, self).__init__(name=name, dtype=dtype)

        # Output depth
        assert output_depth >= 1
        self._output_depth = int(output_depth)

        # Encoder options
        assert isinstance(encoder_options, dict)
        encoder_options.update(dict(name="encoder", dtype=self.dtype))
        self._encoder_options = encoder_options

        # Decoder options
        assert isinstance(decoder_options, dict)
        decoder_options.update(
            dict(autoregressive_mode=True, name="decoder", dtype=self.dtype)
        )
        self._decoder_options = decoder_options

        # Output activations
        self._output_activations = output_activations

        # Start token initializer
        assert isinstance(start_token_initializer, str)
        if start_token_initializer not in START_TOKEN_INITIALIZERS:
            raise ValueError(
                "`start_token_initializer` should be selected "
                f"in {START_TOKEN_INITIALIZERS}, instead "
                f"'{start_token_initializer}' passed"
            )
        self._start_token_initializer = start_token_initializer

        # Encoder
        self._encoder = Encoder(**encoder_options)

        # Decoder
        self._decoder = Decoder(**decoder_options)

        # Final layers
        self._output_layer, self._filter = self._prepare_final_layers(
            output_depth=self._output_depth,
            output_activations=self._output_activations,
            dtype=self.dtype,
        )
