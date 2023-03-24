import tensorflow as tf


class BaseLoss:
    def __init__(self, name="loss") -> None:
        if not isinstance(name, str):
            raise TypeError(
                f"`name` should be a string " f"instead {type(name)} passed"
            )
        self._name = name

    def transformer_loss(
        self,
        transformer,
        discriminator,
        source,
        target,
        sample_weight=None,
        training=True,
    ) -> tf.Tensor:
        raise NotImplementedError(
            "Only `BaseLoss` subclasses have the "
            "`transformer_loss()` method implemented."
        )

    def discriminator_loss(
        self,
        transformer,
        discriminator,
        source,
        target,
        sample_weight=None,
        training=True,
    ) -> tf.Tensor:
        raise NotImplementedError(
            "Only `BaseLoss` subclasses have the "
            "`discriminator_loss()` method implemented."
        )

    def aux_classifier_loss(
        self, aux_classifier, source, target, sample_weight=None, training=True
    ) -> tf.Tensor:
        raise NotImplementedError(
            "Only some `BaseLoss` subclasses have the "
            "`aux_classifier_loss()` method implemented."
        )

    @property
    def name(self) -> str:
        return self._name
