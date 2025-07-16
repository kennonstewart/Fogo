"""
Fogo: Online Gradient Boosted Decision Trees for Edge Learning and Machine Unlearning

This package provides a standalone memory pair software package that can be cloned
into another repository to perform machine unlearning testing.
"""

from .memory_pair import StreamNewtonMemoryPair
from .l_bfgs import LimitedMemoryBFGS, OnlineLBFGS
from .event_logging import init_logging

__version__ = "0.1.0"
__all__ = [
    "StreamNewtonMemoryPair",
    "LimitedMemoryBFGS", 
    "OnlineLBFGS",
    "init_logging"
]