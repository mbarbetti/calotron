import tensorflow as tf

from calotron.callbacks.schedulers.AdvBaseDamping import AdvBaseDamping


class AdvExpDamping(AdvBaseDamping):
    def __init__(
        self,
        adv_scale,
        decay_rate,
        decay_steps,
        staircase=False,
        min_adv_scale=None,
        verbose=False,
    ) -> None:
        super().__init__(adv_scale, verbose)

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

    def on_train_begin(self, logs=None) -> None:
        super().on_train_begin(logs=logs)
        self._tf_decay_rate = tf.cast(self._decay_rate, self._dtype)
        self._tf_decay_steps = tf.cast(self._decay_steps, self._dtype)

    def _scheduled_scale(self, init_scale, step) -> tf.Tensor:
        p = tf.divide(step, self._tf_decay_steps)
        if self._staircase:
            p = tf.floor(p)
        sched_scale = tf.multiply(init_scale, tf.pow(self._tf_decay_rate, p))
        if self._min_adv_scale is not None:
            return tf.maximum(sched_scale, self._min_adv_scale)
        else:
            return sched_scale

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
