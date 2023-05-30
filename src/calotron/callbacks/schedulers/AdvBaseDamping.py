import tensorflow as tf
from tensorflow.keras.callbacks import Callback


class AdvBaseDamping(Callback):
    def __init__(self, adv_scale, verbose=False) -> None:
        super().__init__()

        # Adversarial scale
        assert isinstance(adv_scale, tf.Variable)
        self._adv_scale = adv_scale

        # Verbose
        assert isinstance(verbose, bool)
        self._verbose = verbose

    def on_train_begin(self, logs=None) -> None:
        self._init_adv_scale = tf.identity(self._adv_scale)
        self._dtype = self._init_adv_scale.dtype
        self._step = -1

    def on_batch_begin(self, batch, logs=None) -> None:
        self._step += 1
        step = tf.cast(self._step, self._dtype)
        self._adv_scale.assign(self._scheduled_scale(self._init_adv_scale, step))

    def _scheduled_scale(self, init_scale, step) -> tf.Tensor:
        return init_scale

    def on_batch_end(self, batch, logs=None) -> None:
        logs = logs or {}
        if self._verbose:
            key, _ = self._adv_scale.name.split(":")
            logs[key] = self._adv_scale.numpy()

    def on_epoch_end(self, epoch, logs=None) -> None:
        logs = logs or {}
        if self._verbose:
            key, _ = self._adv_scale.name.split(":")
            logs[key] = self._adv_scale.numpy()

    @property
    def adv_scale(self) -> tf.Variable:
        return self._adv_scale
