import tensorflow as tf

from calotron.callbacks.schedulers.LearnRateBaseScheduler import LearnRateBaseScheduler


class LearnRateAttnDecay(LearnRateBaseScheduler):
    def __init__(self, optimizer, d_model, warmup_steps=4000, verbose=False) -> None:
        super().__init__(optimizer, verbose)

        # d_model
        assert isinstance(d_model, (int, float))
        assert d_model >= 1
        self._d_model = int(d_model)

        # Warmup steps
        assert isinstance(warmup_steps, (int, float))
        assert warmup_steps >= 1
        self._warmup_steps = int(warmup_steps)

    def on_train_begin(self, logs=None) -> None:
        super().on_train_begin(logs=logs)
        self._tf_d_model = tf.cast(self._d_model, self._dtype)
        self._tf_warmup_steps = tf.cast(self._warmup_steps, self._dtype)

    def _scheduled_lr(self, init_lr, step) -> tf.Tensor:
        arg1 = tf.math.rsqrt(step)
        arg2 = tf.multiply(step, tf.pow(self._tf_warmup_steps, -1.5))
        return tf.multiply(tf.math.rsqrt(self._tf_d_model), tf.math.minimum(arg1, arg2))

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def warmup_steps(self) -> int:
        return self._warmup_steps
