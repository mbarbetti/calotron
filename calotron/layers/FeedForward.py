import tensorflow as tf


class FeedForward(tf.keras.layers.Layer):
  def __init__(self, output_units, hidden_units, dropout_rate=0.1):
    super().__init__()
    self._output_units = int(output_units)
    self._hidden_units = int(hidden_units)
    self._dropout_rate = float(dropout_rate)

    self._emb_layer = tf.keras.layers.Dense(self._output_units)

    self._seq = tf.keras.Sequential([
        tf.keras.layers.Dense(self._hidden_units, activation="relu"),
        tf.keras.layers.Dropout(self._dropout_rate),
        tf.keras.layers.Dense(self._output_units)])

    self._add = tf.keras.layers.Add()
    self._layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    x = self._emb_layer(x)
    x = self._add([x, self._seq(x)])
    x = self._layer_norm(x) 
    return x   # shape (batch_size, x_elements, output_units)

  @property
  def output_units(self) -> int:
    return self._output_units

  @property
  def hidden_units(self) -> int:
    return self._hidden_units

  @property
  def dropout_rate(self) -> float:
    return self._dropout_rate
