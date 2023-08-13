import os

import tensorflow as tf
from tensorflow.keras.layers import Add

from calotron.models.players.Encoder import Encoder


class PretrainedEncoder(Encoder):
    def __init__(
        self,
        output_depth,
        num_layers,
        num_heads,
        key_dim,
        admin_res_scale="O(n)",
        mlp_units=128,
        dropout_rate=0.0,
        seq_ord_latent_dim=16,
        seq_ord_max_length=512,
        seq_ord_normalization=10_000,
        enable_res_smoothing=True,
        pretrained_model_dir=None,
        name=None,
        dtype=None,
    ) -> None:
        super().__init__(
            output_depth=output_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            key_dim=key_dim,
            admin_res_scale=admin_res_scale,
            mlp_units=mlp_units,
            dropout_rate=dropout_rate,
            seq_ord_latent_dim=seq_ord_latent_dim,
            seq_ord_max_length=seq_ord_max_length,
            seq_ord_normalization=seq_ord_normalization,
            enable_res_smoothing=enable_res_smoothing,
            name=name,
            dtype=dtype,
        )

        if pretrained_model_dir is not None:
            assert isinstance(pretrained_model_dir, str)
            if not os.path.exists(pretrained_model_dir):
                raise ValueError(
                    "The directory passed for the encoder model "
                    f"({pretrained_model_dir}) doesn't exist"
                )
            self._pretrained_model_dir = pretrained_model_dir
            self._pretrained_model = tf.keras.models.load_model(pretrained_model_dir)
            self._add = Add()
        else:
            self._pretrained_model_dir = None
            self._pretrained_model = None
            self._add = None

    def call(self, x) -> tf.Tensor:
        if self._pretrained_model is not None:
            out = self._seq_ord_embed(x)
            if self._smooth_seq is not None:
                for layer in self._smooth_seq:
                    out = layer(out)
            if self._pretrained_model is not None:
                pretrain_out = self._pretrained_model(x)
                out = self._add([out, pretrain_out])
            for i in range(self._num_layers):
                out = self._enc_layers[i](out)
            return out
        else:
            return super().call(x)

    @property
    def pretrained_model_dir(self):  # TODO: add Union[str, None]
        return self._pretrained_model_dir
