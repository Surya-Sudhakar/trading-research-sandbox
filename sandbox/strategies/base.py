from __future__ import annotations
from typing import Iterable,Protocol,runtime_checkable
from .context import StrategyContext
from .models import StrategyMetadata,StrategySignal
@runtime_checkable
class StrategyPlugin(Protocol):
    metadata:StrategyMetadata
    def reset(self)->None:...
    def evaluate(self,context:StrategyContext)->Iterable[StrategySignal]:...
