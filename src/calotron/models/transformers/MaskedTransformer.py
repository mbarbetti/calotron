import numpy as np
import tensorflow as tf

from calotron.models.transformers.Transformer import Transformer


def prepare_attention_mask(
    source_len, target_len, init_nonzero_size=1, dtype=tf.float32
) -> tf.Tensor:
    attn_mask = np.zeros(shape=(target_len, source_len))
    ones_mask = np.ones(shape=(source_len,))

    i = 0
    nonzero_size = init_nonzero_size
    while i < target_len:
        left_side = int(nonzero_size * target_len / (2 * source_len))
        right_side = nonzero_size - left_side - 1
        if i < left_side:
            attn_mask[i, :nonzero_size] = ones_mask[:nonzero_size]
        elif (i >= left_side) and (i + right_side < source_len):
            attn_mask[i, (i - left_side) : (i + right_side + 1)] = ones_mask[
                :nonzero_size
            ]
        else:
            attn_mask[i, (i - left_side) :] = ones_mask[(i - left_side) :]
        if i + right_side < source_len:
            nonzero_size += int((source_len - i - right_side) / (target_len - i))
        i += 1

    return tf.cast(attn_mask, dtype=dtype)


class MaskedTransformer(Transformer):
    def __init__(
        self,
        output_depth,
        encoder_depth,
        decoder_depth,
        num_layers,
        num_heads,
        key_dims,
        mlp_units=128,
        dropout_rates=0.1,
        seq_ord_latent_dims=16,
        seq_ord_max_lengths=512,
        seq_ord_normalizations=10_000,
        attn_mask_init_nonzero_size=8,
        enable_residual_smoothing=True,
        enable_source_baseline=True,
        enable_attention_mask=True,
        output_activations=None,
        start_token_initializer="ones",
        name=None,
        dtype=None,
    ) -> None:
        super().__init__(
            output_depth=output_depth,
            encoder_depth=encoder_depth,
            decoder_depth=decoder_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            key_dims=key_dims,
            mlp_units=mlp_units,
            dropout_rates=dropout_rates,
            seq_ord_latent_dims=seq_ord_latent_dims,
            seq_ord_max_lengths=seq_ord_max_lengths,
            seq_ord_normalizations=seq_ord_normalizations,
            enable_residual_smoothing=enable_residual_smoothing,
            enable_source_baseline=enable_source_baseline,
            output_activations=output_activations,
            start_token_initializer=start_token_initializer,
            name=name,
            dtype=dtype,
        )

        # Attention mask init nonzero number (global/causal/cross attn layers)
        if isinstance(attn_mask_init_nonzero_size, (int, float)):
            assert attn_mask_init_nonzero_size >= 1
            self._attn_mask_init_nonzero_size = [int(attn_mask_init_nonzero_size)] * 3
        else:
            assert isinstance(attn_mask_init_nonzero_size, (list, tuple, np.ndarray))
            assert len(attn_mask_init_nonzero_size) == 3
            self._attn_mask_init_nonzero_size = list()
            for nonzero_len in attn_mask_init_nonzero_size:
                assert isinstance(nonzero_len, (int, float))
                self._attn_mask_init_nonzero_size.append(int(nonzero_len))

        # Enable attention mask (global/causal/cross attn layers)
        if isinstance(enable_attention_mask, bool):
            assert isinstance(enable_attention_mask, bool)
            self._enable_attention_mask = [enable_attention_mask] * 3
        else:
            assert isinstance(enable_attention_mask, (list, tuple, np.ndarray))
            assert len(enable_attention_mask) == 3
            self._enable_attention_mask = list()
            for flag in enable_attention_mask:
                assert isinstance(flag, bool)
                self._enable_attention_mask.append(flag)

        # Attention masks
        if self._enable_attention_mask[0]:
            self._global_attn_mask = prepare_attention_mask(
                source_len=self._seq_ord_max_lengths[0],
                target_len=self._seq_ord_max_lengths[0],
                init_nonzero_size=self._attn_mask_init_nonzero_size[0],
                dtype=self.dtype,
            )
        else:
            self._global_attn_mask = None

        if self._enable_attention_mask[1]:
            self._causal_attn_mask = prepare_attention_mask(
                source_len=self._seq_ord_max_lengths[1],
                target_len=self._seq_ord_max_lengths[1],
                init_nonzero_size=self._attn_mask_init_nonzero_size[1],
                dtype=self.dtype,
            )
        else:
            self._causal_attn_mask = None

        if self._enable_attention_mask[2]:
            self._cross_attn_mask = prepare_attention_mask(
                source_len=self._seq_ord_max_lengths[0],
                target_len=self._seq_ord_max_lengths[1],
                init_nonzero_size=self._attn_mask_init_nonzero_size[2],
                dtype=self.dtype,
            )
        else:
            self._cross_attn_mask = None

    def call(self, inputs) -> tf.Tensor:
        source, target = inputs
        target = self._prepare_input_target(target)

        # Masked encoder
        if self._global_attn_mask is not None:
            global_attn_mask = tf.tile(
                self._global_attn_mask[
                    None, : tf.shape(source)[1], : tf.shape(source)[1]
                ],
                multiples=(tf.shape(source)[0], 1, 1),
            )
        else:
            global_attn_mask = self._global_attn_mask
        condition = self._encoder(source, global_attn_mask=global_attn_mask)

        # Masked decoder
        if self._causal_attn_mask is not None:
            causal_attn_mask = tf.tile(
                self._causal_attn_mask[
                    None, : tf.shape(target)[1], : tf.shape(target)[1]
                ],
                multiples=(tf.shape(target)[0], 1, 1),
            )
        else:
            causal_attn_mask = self._causal_attn_mask
        if self._cross_attn_mask is not None:
            cross_attn_mask = tf.tile(
                self._cross_attn_mask[
                    None, : tf.shape(target)[1], : tf.shape(source)[1]
                ],
                multiples=(tf.shape(target)[0], 1, 1),
            )
        else:
            cross_attn_mask = self._cross_attn_mask
        output = self._decoder(
            target,
            condition,
            causal_attn_mask=causal_attn_mask,
            cross_attn_mask=cross_attn_mask,
        )

        # Output layers
        output = self._output_layer(output)
        if self._multi_activations is not None:
            output = self._multi_activations(output)
        baseline = self._prepare_output_baseline(source, target)
        return baseline + output

    @property
    def attn_mask_init_nonzero_size(self) -> list:
        return self._attn_mask_init_nonzero_size

    @property
    def enable_attention_mask(self) -> list:
        return self._enable_attention_mask
