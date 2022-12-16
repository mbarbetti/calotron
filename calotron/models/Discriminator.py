import tensorflow as tf
from calotron.layers import DeepSets


class Discriminator(tf.keras.Model):
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

    self._deepsets = DeepSets(output_dim=self._output_dim,
                              latent_dim=self._latent_dim,
                              hidden_layers=self._hidden_layers,
                              hidden_units=self._hidden_units,
                              dropout_rate=self._dropout_rate)

  def call(self, x):
    return self._deepsets(x)

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

  @property
  def deepsets(self) -> DeepSets:
    return self._deepsets
