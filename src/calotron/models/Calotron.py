import tensorflow as tf

from calotron.models.discriminators import BaseDiscriminator
from calotron.models.transformers import BaseTransformer
from calotron.utils.checks import checkLoss, checkMetrics, checkOptimizer


class Calotron(tf.keras.Model):
    def __init__(self, transformer, discriminator, name=None, dtype=None) -> None:
        super().__init__(name=name, dtype=dtype)

        # Transformer
        if not isinstance(transformer, BaseTransformer):
            raise TypeError(
                f"`transformer` should be a calotron's `BaseTransformer`, "
                f"instead {type(transformer)} passed"
            )
        self._transformer = transformer

        # Discriminator
        if not isinstance(discriminator, BaseDiscriminator):
            raise TypeError(
                f"`discriminator` should be a calotron's `BaseDiscriminator`, "
                f"instead {type(discriminator)} passed"
            )
        self._discriminator = discriminator

    def call(self, inputs) -> tuple:
        source, target = inputs
        output = self._transformer((source, target))
        d_output_true = self._discriminator((source, target))
        d_output_pred = self._discriminator((source, output))
        return output, d_output_true, d_output_pred

    def summary(self, **kwargs) -> None:
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
    ) -> None:
        super().compile(weighted_metrics=[])

        # Loss metrics
        self._loss = checkLoss(loss)
        self._t_loss = tf.keras.metrics.Mean(name="t_loss")
        self._d_loss = tf.keras.metrics.Mean(name="d_loss")
        self._metrics = checkMetrics(metrics)

        # Optimizers
        self._t_opt = checkOptimizer(transformer_optimizer)
        self._d_opt = checkOptimizer(discriminator_optimizer)

        # Transformer updates per batch
        assert isinstance(transformer_upds_per_batch, (int, float))
        assert transformer_upds_per_batch >= 1
        self._t_upds_per_batch = int(transformer_upds_per_batch)

        # Discriminator updates per batch
        assert isinstance(discriminator_upds_per_batch, (int, float))
        assert discriminator_upds_per_batch >= 1
        self._d_upds_per_batch = int(discriminator_upds_per_batch)

    def train_step(self, data) -> dict:
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
            output = self._transformer((source, target), training=False)
            y_pred = self._discriminator(
                (source, output), filter=sample_weight, training=False
            )
            y_true = self._discriminator(
                (source, target), filter=sample_weight, training=False
            )
            for metric in self._metrics:
                metric.update_state(
                    y_true=y_true, y_pred=y_pred, sample_weight=sample_weight
                )
                train_dict.update({metric.name: metric.result()})

        train_dict.update(
            {
                "t_loss": self._t_loss.result(),
                "d_loss": self._d_loss.result(),
                "t_lr": self._t_opt.learning_rate,
                "d_lr": self._d_opt.learning_rate,
            }
        )
        return train_dict

    def _t_train_step(self, source, target, sample_weight) -> None:
        with tf.GradientTape() as tape:
            loss = self._loss.transformer_loss(
                transformer=self._transformer,
                discriminator=self._discriminator,
                source=source,
                target=target,
                sample_weight=sample_weight,
                training=True,
            )
        trainable_vars = self._transformer.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self._t_opt.apply_gradients(zip(gradients, trainable_vars))
        self._t_loss.update_state(loss)

    def _d_train_step(self, source, target, sample_weight) -> None:
        with tf.GradientTape() as tape:
            loss = self._loss.discriminator_loss(
                transformer=self._transformer,
                discriminator=self._discriminator,
                source=source,
                target=target,
                sample_weight=sample_weight,
                training=True,
            )
        trainable_vars = self._discriminator.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self._d_opt.apply_gradients(zip(gradients, trainable_vars))
        self._d_loss.update_state(loss)

    def test_step(self, data) -> dict:
        if len(data) == 3:
            source, target, sample_weight = data
        else:
            source, target = data
            sample_weight = None

        t_loss = self._loss.transformer_loss(
            transformer=self._transformer,
            discriminator=self._discriminator,
            source=source,
            target=target,
            sample_weight=sample_weight,
            training=False,
        )
        self._t_loss.update_state(t_loss)

        d_loss = self._loss.discriminator_loss(
            transformer=self._transformer,
            discriminator=self._discriminator,
            source=source,
            target=target,
            sample_weight=sample_weight,
            training=False,
        )
        self._d_loss.update_state(d_loss)

        train_dict = dict()
        if self._metrics is not None:
            output = self._transformer((source, target), training=False)
            y_pred = self._discriminator(
                (source, output), filter=sample_weight, training=False
            )
            y_true = self._discriminator(
                (source, target), filter=sample_weight, training=False
            )
            for metric in self._metrics:
                metric.update_state(
                    y_true=y_true, y_pred=y_pred, sample_weight=sample_weight
                )
                train_dict.update({metric.name: metric.result()})

        train_dict.update(
            {
                "t_loss": self._t_loss.result(),
                "d_loss": self._d_loss.result(),
                "t_lr": self._t_opt.learning_rate,
                "d_lr": self._d_opt.learning_rate,
            }
        )
        return train_dict

    def get_start_token(self, target) -> tf.Tensor:
        return self._transformer.get_start_token(target)

    @property
    def transformer(self) -> BaseTransformer:
        return self._transformer

    @property
    def discriminator(self) -> BaseDiscriminator:
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
