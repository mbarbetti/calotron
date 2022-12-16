import tensorflow as tf


class LatentMapLayer(tf.keras.layers.Layer):
  def __init__(self, latent_dim, num_layers, hidden_units=128, dropout_rate=0.1):
    super().__init__()
    self._latent_dim = int(latent_dim)
    self._num_layers = int(num_layers)
    self._hidden_units = int(hidden_units)
    self._dropout_rate = float(dropout_rate)

    self._seq = list()
    for _ in range(self._num_layers - 1):
      self._seq.append(tf.keras.layers.Dense(self._hidden_units, activation="relu"))
      self._seq.append(tf.keras.layers.Dropout(self._dropout_rate))
    self._seq += [
        tf.keras.layers.Dense(self._latent_dim, activation="relu")]
  
  def call(self, x):
    # shape: (batch_size, x_elements, x_depth)
    outputs = list()
    for i in range(x.shape[1]):
      latent_tensor = x[:, i:i+1, :]   # shape: (batch_size, 1, x_depth)
      for layer in self._seq:
        latent_tensor = layer(latent_tensor)   # shape: (batch_size, 1, latent_dim)
      outputs.append(latent_tensor)

    concat = tf.keras.layers.Concatenate(axis=1)(outputs)   # shape: (batch_size, x_elements, latent_dim)
    output = tf.reduce_sum(concat, axis=1)   # shape: (batch_size, latent_dim)
    return output

  @property
  def latent_dim(self) -> int:
    return self._latent_dim

  @property
  def num_layers(self) -> int:
    return self._latent_dim

  @property
  def hidden_units(self) -> int:
    return self._hidden_units

  @property
  def dropout_rate(self) -> float:
    return self._dropout_rate


class DeepSets(tf.keras.layers.Layer):
  def __init__(self,
               output_dim,
               latent_dim,
               hidden_layers=3,
               hidden_units=128,
               dropout_rate=0.1):
    super().__init__()
    self._output_dim = int(output_dim)
    self._latent_dim = int(latent_dim)
    self._hidden_layers = int(hidden_layers)
    self._hidden_units = int(hidden_units)
    self._dropout_rate = float(dropout_rate)

    self._latent_map = LatentMapLayer(
        latent_dim=self._latent_dim,
        num_layers=self._hidden_layers + 1,
        hidden_units=self._hidden_units,
        dropout_rate=self._dropout_rate)

    self._seq = [
        tf.keras.layers.Dense(self._latent_dim, activation="relu"),
        tf.keras.layers.Dropout(self._dropout_rate),
        tf.keras.layers.Dense(self._latent_dim, activation="relu"),
        tf.keras.layers.Dropout(self._dropout_rate)]
    self._seq += [tf.keras.layers.Dense(self._output_dim)]

  def call(self, x):
    x = self._latent_map(x)
    for layer in self._seq:
      x = layer(x)
    return x

  @property
  def output_dim(self) -> int:
    return self._output_dim

  @property
  def latent_dim(self) -> int:
    return self._latent_dim

  @property
  def hidden_layers(self) -> int:
    return self._hidden_layers

  @property
  def hidden_units(self) -> int:
    return self._hidden_units

  @property
  def dropout_rate(self) -> float:
    return self._dropout_rate
