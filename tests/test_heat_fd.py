import importlib

def test_pinn_imports():
    module = importlib.import_module("demos.heat_pinn")
    assert module is not None

def test_pinn_has_main():
    module = importlib.import_module("demos.heat_pinn")
    assert hasattr(module, "__file__")
