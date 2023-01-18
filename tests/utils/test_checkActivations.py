import pytest
import tensorflow as tf
from tensorflow.keras.layers import Layer, Activation, ReLU
from tensorflow.keras.activations import sigmoid, tanh, relu


OUTPUT_DIM = 3
STR_CASES = ["sigmoid", "tanh", "relu"]
CLS_CASES = [Activation(sigmoid), Activation(tanh), Activation(relu)]


@pytest.fixture
def checker():
  from calotron.utils import checkActivations
  chk = checkActivations
  return chk


###########################################################################


def test_checker_use_None(checker):
  res = checker(None, OUTPUT_DIM)
  if res is not None:
    assert False


def test_checker_use_string(checker):
  res = checker("relu", OUTPUT_DIM)
  assert isinstance(res, list)
  assert len(res) == OUTPUT_DIM
  for r in res:
    assert isinstance(r, Activation)


def test_checker_use_class(checker):
  res = checker(ReLU(), OUTPUT_DIM)
  assert isinstance(res, list)
  assert len(res) == OUTPUT_DIM
  for r in res:
    assert isinstance(r, Layer)


def test_checker_use_strings(checker):
  res = checker(STR_CASES, OUTPUT_DIM)
  assert isinstance(res, list)
  assert len(res) == OUTPUT_DIM
  for r in res:
    assert isinstance(r, Activation)


def test_checker_use_classes(checker):
  res = checker(CLS_CASES, OUTPUT_DIM)
  assert isinstance(res, list)
  assert len(res) == OUTPUT_DIM
  for r in res:
    assert isinstance(r, Activation)


def test_checker_use_mixture(checker):
  res = checker(STR_CASES + CLS_CASES, 2 * OUTPUT_DIM)
  assert isinstance(res, list)
  assert len(res) == len(STR_CASES) + len(CLS_CASES)
  for r in res:
    assert isinstance(r, Activation)
