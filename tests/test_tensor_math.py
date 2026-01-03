import numpy as np
import importlib

def test_tensor_math_imports():
    module = importlib.import_module("demos.tensor_math")
    assert module is not None

def test_tensor_operations():
    # simple tensor math check
    a = np.array([1, 2, 3])
    b = np.array([3, 2, 1])
    dot = np.dot(a, b)
    assert dot == 10
