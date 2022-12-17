import tensorflow as tf
from calotron.models import Transformer, Discriminator
from calotron.utils import checkOptimizer


class CaloTron(tf.keras.Model):
  def __init__(self, transformer, discriminator):
    super().__init__()
    if not isinstance(transformer, Transformer):
      raise TypeError(f"`transformer` should be a calotron's "
                      f"`Transformer`, instead "
                      f"{type(transformer)} passed")
    self._transformer = transformer
    if not isinstance(discriminator, Discriminator):
      raise TypeError(f"`discriminator` should be a calotron's "
                      f"`Discriminator`, instead "
                      f"{type(discriminator)} passed")
    self._discriminator = discriminator

  def call(self, inputs):
    source, target = inputs
    output = self._transformer((source, target))
    d_output_true = self._discriminator(target)
    d_output_pred = self._discriminator(output)
    return output, d_output_true, d_output_pred

  def summary(self):
    self._transformer.summary()
    self._discriminator.summary()

  def compile(self,
              loss,
              metric=None,
              transformer_optimizer="rmsprop",
              discriminator_optimizer="rmsprop"):
    super().compile()
    self._loss = loss
    self._t_loss_tracker = tf.keras.metrics.Mean(name="t_loss")
    self._d_loss_tracker = tf.keras.metrics.Mean(name="d_loss")
    self._metric = metric
    if metric is not None:
      self._metric_tracker = tf.keras.metrics.MeanSquaredError(name="mse")
    else:
      self._metric_tracker = None
    self._t_opt = checkOptimizer(transformer_optimizer)
    self._d_opt = checkOptimizer(discriminator_optimizer)

  def train_step(self, data):
    self._d_train_step(data)
    self._t_train_step(data)
    train_dict = dict()
    if self._metric is not None:
      train_dict.update({self._metric_tracker.name: self._metric_tracker.result()})
    train_dict.update({"t_loss": self._t_loss_tracker.result(),
                       "d_loss": self._d_loss_tracker.result()})
    return train_dict

  def _d_train_step(self, data):
    source, target = data
    target_in, target_out = self._prepare_target(target)
    with tf.GradientTape() as tape:
      target_pred = self._transformer((source, target_in))
      y_pred = self._discriminator(target_pred)
      y_true = self._discriminator(target_out)
      loss = - self._loss(y_true, y_pred)   # maximize loss
    self._d_loss_tracker.update_state(loss)
    grads = tape.gradient(loss, self._discriminator.trainable_weights)
    self._d_opt.apply_gradients(zip(grads, self._discriminator.trainable_weights))

  def _t_train_step(self, data):
    source, target = data
    target_in, target_out = self._prepare_target(target)
    with tf.GradientTape() as tape:
      target_pred = self._transformer((source, target_in))
      y_pred = self._discriminator(target_pred)
      y_true = self._discriminator(target_out)
      loss = self._loss(y_true, y_pred)   # minimize loss
    self._t_loss_tracker.update_state(loss)
    grads = tape.gradient(loss, self._transformer.trainable_weights)
    self._t_opt.apply_gradients(zip(grads, self._transformer.trainable_weights))

  @staticmethod
  def _prepare_target(target):
    batch_size = target.shape[0]
    target_depth = target.shape[2]
    start_token = tf.random.normal(shape=(batch_size, 1, target_depth))
    target_in = tf.concat([start_token, target[:, :-1, :]], axis=1)
    target_out = target
    return target_in, target_out

  @property
  def transformer(self) -> Transformer:
    return self._transformer

  @property
  def discriminator(self) -> Discriminator:
    return self._discriminator

  @property
  def metrics(self):
    if self._metric is not None:
      return [self._t_loss_tracker,
              self._d_loss_tracker,
              self._metric_tracker]
    else:
      return [self._t_loss_tracker, self._d_loss_tracker]

  @property
  def transformer_optimizer(self) -> tf.keras.optimizers.Optimizer:
    return self._t_opt

  @property
  def discriminator_optimizer(self) -> tf.keras.optimizers.Optimizer:
    return self._d_opt
