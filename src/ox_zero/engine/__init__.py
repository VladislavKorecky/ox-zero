"""The engine: position analysis behind a stable interface.

For now only a placeholder `DummyEngine` exists; the AlphaZero search
(roadmap step 3) will implement the same `Engine` protocol.
"""

from ox_zero.engine.analysis import Analysis, Engine, analyze
from ox_zero.engine.dummy import DummyEngine, load_engine

__all__ = ["Analysis", "DummyEngine", "Engine", "analyze", "load_engine"]
