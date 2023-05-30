import tensorflow as tf

from calotron.callbacks.schedulers.AdvBaseDamping import AdvBaseDamping


class AdvLinearDamping(AdvBaseDamping):
    def __init__(self, adv_scale, steps_to_min, min_adv_scale, verbose=False) -> None:
        super().__init__(adv_scale, verbose)

        # Steps to min
        assert isinstance(steps_to_min, (int, float))
        assert steps_to_min >= 1
        self._steps_to_min = int(steps_to_min)

        # Minimum adversarial scale
        assert isinstance(min_adv_scale, (int, float))
        assert min_adv_scale > 0.0
        self._min_adv_scale = float(min_adv_scale)

    def on_train_begin(self, logs=None) -> None:
        super().on_train_begin(logs=logs)
        self._tf_steps_to_min = tf.cast(self._steps_to_min, self._dtype)
        self._tf_min_adv_scale = tf.cast(self._min_adv_scale, self._dtype)

    def _scheduled_scale(self, init_scale, step) -> tf.Tensor:
        m = (self._tf_min_adv_scale - init_scale) / self._tf_steps_to_min
        sched_scale = m * step + init_scale
        return tf.maximum(sched_scale, self._min_adv_scale)

    @property
    def steps_to_min(self) -> float:
        return self._steps_to_min

    @property
    def min_adv_scale(self) -> int:
        return self._min_adv_scale
