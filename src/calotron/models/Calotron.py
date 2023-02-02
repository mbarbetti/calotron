import tensorflow as tf

from calotron.models.Discriminator import Discriminator
from calotron.models.Transformer import Transformer
from calotron.utils import checkLoss, checkMetrics, checkOptimizer


class Calotron(tf.keras.Model):
    def __init__(self, transformer, discriminator, name=None, dtype=None):
        super().__init__(name=name, dtype=dtype)
        if not isinstance(transformer, Transformer):
            raise TypeError(
                f"`transformer` should be a calotron's "
                f"`Transformer`, instead "
                f"{type(transformer)} passed"
            )
        self._transformer = transformer
        if not isinstance(discriminator, Discriminator):
            raise TypeError(
                f"`discriminator` should be a calotron's "
                f"`Discriminator`, instead "
                f"{type(discriminator)} passed"
            )
        self._discriminator = discriminator

    def call(self, inputs):
        source, target = inputs
        output = self._transformer((source, target))
        d_output_true = self._discriminator(target)
        d_output_pred = self._discriminator(output)
        return output, d_output_true, d_output_pred

    def summary(self, **kwargs):
        print("_" * 65)
        self._transformer.summary(**kwargs)
        self._discriminator.summary(**kwargs)

    def compile(
        self,
        loss,
        metrics=None,
        transformer_optimizer="rmsprop",
        discriminator_optimizer="rmsprop",
        transformer_upds_per_batch=1,
        discriminator_upds_per_batch=1,
    ):
        super().compile()
        self._loss = checkLoss(loss)
        self._t_loss = tf.keras.metrics.Mean(name=f"t_{self._loss.name}")
        self._d_loss = tf.keras.metrics.Mean(name=f"d_{self._loss.name}")
        self._metrics = checkMetrics(metrics)
        self._t_opt = checkOptimizer(transformer_optimizer)
        self._d_opt = checkOptimizer(discriminator_optimizer)
        if transformer_upds_per_batch < 1:
            raise ValueError("`transformer_upds_per_batch` should be greater than 0")
        self._t_upds_per_batch = int(transformer_upds_per_batch)
        if discriminator_upds_per_batch < 1:
            raise ValueError("`discriminator_upds_per_batch` should be greater than 0")
        self._d_upds_per_batch = int(discriminator_upds_per_batch)

    def train_step(self, data):
        if len(data) == 3:
            source, target, sample_weight = data
        else:
            source, target = data
            sample_weight = None

        for _ in range(self._d_upds_per_batch):
            self._d_train_step(source, target, sample_weight)
        for _ in range(self._t_upds_per_batch):
            self._t_train_step(source, target, sample_weight)

        train_dict = dict()
        if self._metrics is not None:
            for metric in self._metrics:
                train_dict.update({metric.name: metric.result()})
        train_dict.update(
            {
                f"t_{self._loss.name}": self._t_loss.result(),
                f"d_{self._loss.name}": self._d_loss.result(),
                "t_lr": self._t_opt.learning_rate,
                "d_lr": self._d_opt.learning_rate,
            }
        )
        return train_dict

    def _d_train_step(self, source, target, sample_weight):
        with tf.GradientTape() as tape:
            output = self._transformer((source, target), training=False)
            loss = self._loss.discriminator_loss(
                discriminator=self._discriminator,
                target_true=target,
                target_pred=output,
                sample_weight=sample_weight,
                training=True,
            )
        trainable_vars = self._discriminator.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self._d_opt.apply_gradients(zip(gradients, trainable_vars))
        self._d_loss.update_state(loss)

    def _t_train_step(self, source, target, sample_weight):
        with tf.GradientTape() as tape:
            output = self._transformer((source, target), training=True)
            loss = self._loss.transformer_loss(
                discriminator=self._discriminator,
                target_true=target,
                target_pred=output,
                sample_weight=sample_weight,
                training=False,
            )
        trainable_vars = self._transformer.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self._t_opt.apply_gradients(zip(gradients, trainable_vars))
        self._t_loss.update_state(loss)
        if self._metrics is not None:
            y_pred = self._discriminator(output, training=False)
            y_true = self._discriminator(target, training=False)
            for metric in self._metrics:
                metric.update_state(
                    y_true=y_true, y_pred=y_pred, sample_weight=sample_weight
                )

    def get_start_token(self, target) -> tf.Tensor:
        return self._transformer.get_start_token(target)

    @property
    def transformer(self) -> Transformer:
        return self._transformer

    @property
    def discriminator(self) -> Discriminator:
        return self._discriminator

    @property
    def metrics(self) -> list:
        reset_states = [self._t_loss, self._d_loss]
        if self._metrics is not None:
            reset_states += self._metrics
        return reset_states

    @property
    def transformer_optimizer(self) -> tf.keras.optimizers.Optimizer:
        return self._t_opt

    @property
    def discriminator_optimizer(self) -> tf.keras.optimizers.Optimizer:
        return self._d_opt

    @property
    def transformer_upds_per_batch(self) -> int:
        return self._t_upds_per_batch

    @property
    def discriminator_upds_per_batch(self) -> int:
        return self._d_upds_per_batch
