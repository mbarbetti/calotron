import pytest
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations

STR_CASES = ["sigmoid", "tanh", "relu"]
CLS_CASES = [layers.Activation(activations.sigmoid),
             layers.Activation(activations.tanh),
             layers.Activation(activations.relu)]


@pytest.fixture
def layer():
  from calotron.layers import MultiActivations
  ma = MultiActivations(activations="sigmoid",
                        output_depth=3)
  return ma


###########################################################################


@pytest.mark.parametrize("activations", [None, "sigmoid"])
def test_layer_configuration(activations):
  from calotron.layers import MultiActivations
  layer = MultiActivations(activations=activations,
                           output_depth=3)
  assert isinstance(layer, MultiActivations)
  if layer.output_activations is not None:
    assert isinstance(layer.output_activations, list)
  assert isinstance(layer.output_depth, int)


@pytest.mark.parametrize("activations",
                         [None, "relu", layers.ReLU(), STR_CASES, CLS_CASES])
def test_layer_use(activations):
  from calotron.layers import MultiActivations
  layer = MultiActivations(activations=activations,
                           output_depth=3)
  input = tf.keras.Input(shape=[16, 3])
  output = layer(input)
  test_shape = list(input.shape)
  test_shape[-1] = layer.output_depth
  assert output.shape == tuple(test_shape)
