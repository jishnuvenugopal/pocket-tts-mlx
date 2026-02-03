"""State management helpers for streaming modules."""

from __future__ import annotations

from abc import ABC, abstractmethod

import mlx.nn as nn


def init_states(model: nn.Module, batch_size: int, sequence_length: int) -> dict[str, dict]:
    """Collect per-module streaming state for a model."""
    result = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, StatefulModule):
            continue
        module._module_absolute_name = module_name
        module_state = module.init_state(batch_size, sequence_length=sequence_length)
        result[module_name] = module_state
    return result


def increment_steps(module: nn.Module, model_state: dict[str, dict], increment: int = 1):
    """Advance step counters for all stateful modules."""
    for module_name, mod in module.named_modules():
        if not isinstance(mod, StatefulModule):
            continue
        mod.increment_step(model_state[module_name], increment)


class StatefulModule(ABC, nn.Module):
    """Module base class with explicit streaming state initialization."""
    def __init__(self, *args, **kwds):
        self._module_absolute_name = None
        super().__init__(*args, **kwds)

    @abstractmethod
    def init_state(self, batch_size: int, sequence_length: int):
        """Return initial state dict for streaming."""
        raise NotImplementedError

    def increment_step(self, state: dict, increment: int = 1):
        """Advance state counters after each step."""
        pass

    def get_state(self, model_state: dict[str, dict]):
        """Fetch module-specific state from global model state."""
        return model_state[self._module_absolute_name]
