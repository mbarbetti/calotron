class BaseLoss:
  def __init__(self, name="loss"):
    if not isinstance(name, str):
      raise TypeError(f"`name` should be a string "
                      f"instead {type(name)} passed")
    self._name = name
    self._loss = None

  def __call__(self, y_true, y_pred, **kwargs):
    return self._loss(y_true, y_pred, **kwargs)

  def discriminator_loss(self, y_true, y_pred, **kwargs):
    raise NotImplementedError(f"Only `BaseLoss` subclasses have the "
                              f"`discriminator_loss()` method implemented.")

  def transformer_loss(self, y_true, y_pred, **kwargs):
    raise NotImplementedError(f"Only `BaseLoss` subclasses have the "
                              f"`transformer_loss()` method implemented.")

  @property
  def name(self) -> str:
    return self._name
