import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Optimizer

K = tf.keras.backend


class LearnRateBaseScheduler(Callback):
    def __init__(self, optimizer, verbose=False) -> None:
        super().__init__()

        # Optimizer
        assert isinstance(optimizer, Optimizer)
        self._optimizer = optimizer

        # Verbose
        assert isinstance(verbose, bool)
        self._verbose = verbose

    def on_train_begin(self, logs=None) -> None:
        init_lr = K.get_value(self._optimizer.learning_rate)
        self._init_lr = tf.identity(init_lr)
        self._dtype = self._init_lr.dtype
        self._step = -1

    def on_batch_begin(self, batch, logs=None) -> None:
        self._step += 1
        step = tf.cast(self._step, self._dtype)
        K.set_value(
            self._optimizer.learning_rate, self._scheduled_lr(self._init_lr, step)
        )

    def _scheduled_lr(self, init_lr, step) -> tf.Tensor:
        return init_lr

    def on_batch_end(self, batch, logs=None) -> None:
        logs = logs or {}
        if self._verbose:
            logs["lr"] = K.get_value(self.model.optimizer.learning_rate)

    def on_epoch_end(self, epoch, logs=None) -> None:
        logs = logs or {}
        if self._verbose:
            logs["lr"] = K.get_value(self.model.optimizer.learning_rate)

    @property
    def optimizer(self) -> tf.keras.optimizers.Optimizer:
        return self._optimizer
