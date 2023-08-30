import tensorflow as tf
from tensorflow import keras

from calotron.models.discriminators import Discriminator
from calotron.models.transformers import Transformer
from calotron.utils.checks import checkLoss, checkMetrics, checkOptimizer


class Calotron(keras.Model):
    def __init__(self, transformer, discriminator, name=None, dtype=None) -> None:
        super().__init__(name=name, dtype=dtype)

        # Transformer
        if not isinstance(transformer, Transformer):
            raise TypeError(
                f"`transformer` should be a calotron's `Transformer`, "
                f"instead {type(transformer)} passed"
            )
        self._transformer = transformer

        # Discriminator
        if not isinstance(discriminator, Discriminator):
            raise TypeError(
                f"`discriminator` should be a calotron's `Discriminator`, "
                f"instead {type(discriminator)} passed"
            )
        self._discriminator = discriminator

    def call(self, inputs) -> tuple:
        source, target = inputs
        t_out = self._transformer((source, target))
        d_out_pred = self._discriminator((source, t_out))
        d_out_true = self._discriminator((source, target))
        return t_out, d_out_pred, d_out_true

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
        self._t_loss = keras.metrics.Mean(name="t_loss")
        self._d_loss = keras.metrics.Mean(name="d_loss")
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
        source, target, sample_weight = self._unpack_data(data)

        for _ in range(self._d_upds_per_batch):
            self._d_train_step(source, target, sample_weight)
        for _ in range(self._t_upds_per_batch):
            self._t_train_step(source, target, sample_weight)

        train_dict = dict(t_loss=self._t_loss.result(), d_loss=self._d_loss.result())
        if self._metrics is not None:
            t_out = self._transformer((source, target), training=False)
            source_concat = tf.concat([source, source], axis=0)
            target_concat = tf.concat([target, t_out], axis=0)
            if sample_weight is not None:
                mask = tf.cast(sample_weight > 0.0, dtype=target.dtype)
                mask_concat = tf.concat([mask, mask], axis=0)
            else:
                mask_concat = None
            d_out = self._discriminator(
                (source_concat, target_concat), padding_mask=mask_concat, training=False
            )
            y_true, y_pred = tf.split(d_out, 2, axis=0)
            for metric in self._metrics:
                metric.update_state(
                    y_true=y_true, y_pred=y_pred, sample_weight=sample_weight
                )
                train_dict.update({metric.name: metric.result()})
        return train_dict

    @staticmethod
    def _unpack_data(data) -> tuple:
        if len(data) == 3:
            source, target, sample_weight = data
        else:
            source, target = data
            sample_weight = None
        return source, target, sample_weight

    def _t_train_step(self, source, target, sample_weight=None) -> None:
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

    def _t_enc_train_step(self, source, target, sample_weight=None) -> None:
        with tf.GradientTape() as tape:
            loss = self._loss.transformer_loss(
                transformer=self._transformer,
                discriminator=self._discriminator,
                source=source,
                target=target,
                sample_weight=sample_weight,
                training=True,
            )
        trainable_vars = self._transformer._encoder.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self._t_opt.apply_gradients(zip(gradients, trainable_vars))
        self._t_loss.update_state(loss)

    def _d_train_step(self, source, target, sample_weight=None) -> None:
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

    def _d_enc_train_step(self, source, target, sample_weight=None) -> None:
        with tf.GradientTape() as tape:
            loss = self._loss.discriminator_loss(
                transformer=self._transformer,
                discriminator=self._discriminator,
                source=source,
                target=target,
                sample_weight=sample_weight,
                training=True,
            )
        trainable_vars = self._discriminator._encoder.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self._d_opt.apply_gradients(zip(gradients, trainable_vars))
        self._d_loss.update_state(loss)

    def test_step(self, data) -> dict:
        source, target, sample_weight = self._unpack_data(data)

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

        train_dict = dict(t_loss=self._t_loss.result(), d_loss=self._d_loss.result())
        if self._metrics is not None:
            t_out = self._transformer((source, target), training=False)
            source_concat = tf.concat([source, source], axis=0)
            target_concat = tf.concat([target, t_out], axis=0)
            if sample_weight is not None:
                mask = tf.cast(sample_weight > 0.0, dtype=target.dtype)
                mask_concat = tf.concat([mask, mask], axis=0)
            else:
                mask_concat = None
            d_out = self._discriminator(
                (source_concat, target_concat), padding_mask=mask_concat, training=False
            )
            y_true, y_pred = tf.split(d_out, 2, axis=0)
            for metric in self._metrics:
                metric.update_state(
                    y_true=y_true, y_pred=y_pred, sample_weight=sample_weight
                )
                train_dict.update({metric.name: metric.result()})
        return train_dict

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
    def transformer_optimizer(self) -> keras.optimizers.Optimizer:
        return self._t_opt

    @property
    def discriminator_optimizer(self) -> keras.optimizers.Optimizer:
        return self._d_opt

    @property
    def transformer_upds_per_batch(self) -> int:
        return self._t_upds_per_batch

    @property
    def discriminator_upds_per_batch(self) -> int:
        return self._d_upds_per_batch
