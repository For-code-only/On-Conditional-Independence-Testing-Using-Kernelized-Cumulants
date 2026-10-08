"""CSIC, CSIC–CI, HSIC and matched KCI-type tests."""
__version__ = "0.1.0"
__all__ = ["test"]


def __getattr__(name):
    if name == "test":
        from .inference import test
        return test
    raise AttributeError(name)
