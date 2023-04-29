import tensorflow as tf
from tensorflow.keras.callbacks import Callback


class AdvDamping(Callback):
    def __init__(
        self,
        adv_scale,
        decay_rate,
        decay_steps,
        staircase=False,
        min_adv_scale=None,
        verbose=False,
    ) -> None:
        super().__init__()

        # Adversarial scale
        assert isinstance(adv_scale, tf.Variable)
        self._adv_scale = adv_scale

        # Decay rate
        assert isinstance(decay_rate, (int, float))
        assert decay_rate > 0.0
        self._decay_rate = float(decay_rate)

        # Decay steps
        assert isinstance(decay_steps, (int, float))
        assert decay_steps >= 1
        self._decay_steps = int(decay_steps)

        # Staircase
        assert isinstance(staircase, bool)
        self._staircase = staircase

        # Minimum adversarial scale
        if min_adv_scale is not None:
            assert isinstance(min_adv_scale, (int, float))
            assert min_adv_scale > 0.0
            self._min_adv_scale = float(min_adv_scale)
        else:
            self._min_adv_scale = None

        # Verbose
        assert isinstance(verbose, bool)
        self._verbose = verbose

    def on_train_begin(self, logs=None) -> None:
        self._step = -1
        self._dtype = self._adv_scale.dtype
        self._init_adv_scale = tf.identity(self._adv_scale)
        self._tf_decay_rate = tf.cast(self._decay_rate, self._dtype)
        self._tf_decay_steps = tf.cast(self._decay_steps, self._dtype)

    def on_batch_begin(self, batch, logs=None) -> None:
        self._step += 1
        step = tf.cast(self._step, self._dtype)
        self._adv_scale.assign(self._scheduled_scale(self._init_adv_scale, step))

    def _scheduled_scale(self, init_scale, step) -> tf.Tensor:
        p = tf.divide(step, self._tf_decay_steps)
        if self._staircase:
            p = tf.floor(p)
        sched_scale = tf.multiply(init_scale, tf.pow(self._tf_decay_rate, p))
        if self._min_adv_scale is not None:
            return tf.maximum(sched_scale, self._min_adv_scale)
        else:
            return sched_scale

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

    @property
    def decay_rate(self) -> float:
        return self._decay_rate

    @property
    def decay_steps(self) -> int:
        return self._decay_steps

    @property
    def staircase(self) -> bool:
        return self._staircase

    @property
    def min_adv_scale(self) -> float:
        return self._min_adv_scale
