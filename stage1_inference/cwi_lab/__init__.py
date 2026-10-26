"""CPU-only learning demo. Not a production inspection or medical system."""
from .inference import infer, read_tiff, InputError
__all__ = ['infer', 'read_tiff', 'InputError']
