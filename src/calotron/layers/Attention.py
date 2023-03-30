import tensorflow as tf


class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self._add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
    def call(self, x, condition) -> tf.Tensor:
        attn_output = self._mha(query=x, key=condition, value=condition)
        output = self._add([x, attn_output])
        return output


class GlobalSelfAttention(BaseAttention):
    def call(self, x) -> tf.Tensor:
        attn_output = self._mha(query=x, key=x, value=x)
        output = self._add([x, attn_output])
        return output


class CausalSelfAttention(BaseAttention):
    def call(self, x) -> tf.Tensor:
        attn_output = self._mha(query=x, key=x, value=x, use_causal_mask=True)
        output = self._add([x, attn_output])
        return output
