class BaseLoss:
    def __init__(self, name="loss"):
        if not isinstance(name, str):
            raise TypeError(
                f"`name` should be a string " f"instead {type(name)} passed"
            )
        self._name = name
        self._loss = None

    def discriminator_loss(
        self, discriminator, target_true, target_pred, sample_weight=None
    ):
        raise NotImplementedError(
            "Only `BaseLoss` subclasses have the "
            "`discriminator_loss()` method implemented."
        )

    def transformer_loss(
        self, discriminator, target_true, target_pred, sample_weight=None
    ):
        raise NotImplementedError(
            "Only `BaseLoss` subclasses have the "
            "`transformer_loss()` method implemented."
        )

    @property
    def name(self) -> str:
        return self._name
