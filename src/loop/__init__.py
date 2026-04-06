"""
src.loop — Live intraday paper-trading loop.

Public API
----------
LiveLoop       : main loop class; call .run(series) -> LiveLoopResult
LoopConfig     : dataclass of loop parameters
LiveLoopResult : structured result from .run()
"""

from src.loop.config import LoopConfig
from src.loop.engine import LiveLoop
from src.loop.result import LiveLoopResult

__all__ = ["LiveLoop", "LoopConfig", "LiveLoopResult"]
