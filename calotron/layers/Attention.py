import tensorflow as tf


class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self._mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self._layernorm = tf.keras.layers.LayerNormalization()
    self._add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
  def call(self, x, context):   # x -> target, context -> source
    attn_output = self._mha(
        query=x,
        key=context,
        value=context)
    x = self._add([x, attn_output])
    x = self._layernorm(x)
    return x   # shape: (batch_size, x_elements, x_depth)


class GlobalSelfAttention(BaseAttention):
  def call(self, x):   # x -> source
    attn_output = self._mha(
        query=x,
        key=x,
        value=x)
    x = self._add([x, attn_output])
    x = self._layernorm(x)
    return x   # shape: (batch_size, x_elements, x_depth)


class CausalSelfAttention(BaseAttention):
  def call(self, x):   # x -> target
    attn_output = self._mha(
        query=x,
        key=x,
        value=x,
        use_causal_mask=True)
    x = self._add([x, attn_output])
    x = self._layernorm(x)
    return x   # shape: (batch_size, x_elements, x_depth)
