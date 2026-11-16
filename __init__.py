"""DINO-WT. Heavy imports (PyTorch) are lazy so that `dinowt.metrics` works without torch."""
__version__ = "1.0.0"


def __getattr__(name):
    if name in ("DINOWT", "build_model"):
        from . import models
        return getattr(models, name)
    raise AttributeError(name)
