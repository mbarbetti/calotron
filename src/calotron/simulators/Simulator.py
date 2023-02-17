import tensorflow as tf

from calotron.models import Transformer


class Simulator(tf.Module):
    def __init__(self, transformer, start_token, name=None):
        super().__init__(name=name)
        if not isinstance(transformer, Transformer):
            raise TypeError(
                "`transformer` should be a calotron's `Transformer`, "
                f"instead {type(transformer)} passed"
            )
        self._transformer = transformer
        self._dtype = self._transformer.dtype

        start_token = tf.convert_to_tensor(start_token, dtype=self._dtype)
        if len(start_token.shape) > 2:
            raise ValueError(
                "`start_token` shape should match with "
                "(batch_size, target_depth) or (target_depth,)"
            )
        if start_token.shape[-1] != self._transformer.output_depth:
            raise ValueError(
                "`start_token` elements should match with "
                "the `transformer` output depth, instead "
                f"{start_token.shape[-1]} passed"
            )
        self._start_token = start_token

    def __call__(self, source, max_length):
        source = tf.convert_to_tensor(source, dtype=self._dtype)
        assert max_length >= 1
        max_length = int(max_length)

        if len(self._start_token.shape) == 1:
            start_token = self._start_token[None, :]
            start_token = tf.tile(start_token, (source.shape[0], 1))
            target = tf.expand_dims(start_token, axis=1)
        else:
            if source.shape[0] != self._start_token.shape[0]:
                raise ValueError(
                    "`source` and `start_token` batch-sizes should "
                    f"match, instead {source.shape[0]} and "
                    f"{self._start_token.shape[0]} passed"
                )
            else:
                target = tf.expand_dims(self._start_token, axis=1)
        for _ in tf.range(max_length):
            predictions = self.transformer((source, target), training=False)
            target = tf.concat([target, predictions[:, -1:, :]], axis=1)

        assert target.shape[1] == max_length + 1
        return target[:, 1:, :]

    @property
    def transformer(self) -> Transformer:
        return self._transformer

    @property
    def start_token(self) -> tf.Tensor:
        return self._start_token
