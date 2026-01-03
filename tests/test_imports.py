import importlib

DEMO_MODULES = [
    "demos.heat_fd",
    "demos.heat_pinn",
    "demos.compare_fd_pinn",
    "demos.tensor_math",
    "demos.demo_2_tensors",
    "demos.Tensor_demo",
]

def test_demo_imports():
    for module_name in DEMO_MODULES:
        module = importlib.import_module(module_name)
        assert module is not None
