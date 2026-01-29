"""MLX Stateful Module base class for streaming inference.

This module provides the base class for stateful modules in MLX, mirroring
the PyTorch implementation but adapted for MLX's functional programming model.

Key differences from PyTorch:
- MLX arrays are immutable; use mx.update() for efficient state updates
- State is managed explicitly through state dictionaries rather than mutable buffers
- The init_states and increment_steps functions work with MLX arrays
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import mlx.core as mx
import mlx.nn as nn


def init_states(
    model: nn.Module, batch_size: int, sequence_length: int
) -> Dict[str, Dict[str, mx.array]]:
    """Initialize states for all stateful modules in the model.

    Args:
        model: The MLX module containing stateful submodules.
        batch_size: Batch size for the state tensors.
        sequence_length: Maximum sequence length for KV caches.

    Returns:
        Dictionary mapping module names to their state dictionaries.
        Each state dictionary contains MLX arrays for module-specific state.
    """
    result = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, StatefulModule):
            continue
        module._module_absolute_name = module_name
        module_state = module.init_state(batch_size, sequence_length=sequence_length)
        result[module_name] = module_state
    return result


def increment_steps(
    module: nn.Module, model_state: Dict[str, Dict[str, mx.array]], increment: int = 1
) -> None:
    """Increment the step counter for all stateful modules.

    Args:
        module: The MLX module containing stateful submodules.
        model_state: Dictionary mapping module names to their state dictionaries.
        increment: Number of steps to increment by.
    """
    for module_name, module in module.named_modules():
        if not isinstance(module, StatefulModule):
            continue
        module.increment_step(model_state[module_name], increment)


class StatefulModule(ABC, nn.Module):
    """Base class for stateful modules in MLX.

    Stateful modules maintain internal state across forward passes, enabling
    streaming inference with KV caches and other incremental computations.

    Key methods:
        init_state: Initialize the state dictionary for this module.
        increment_step: Update state to advance by the given number of steps.
        get_state: Extract this module's state from the full model state.

    Note:
        MLX arrays are immutable. State updates should use mx.update() to
        efficiently create new arrays with updated values.
    """

    def __init__(self, *args, **kwargs):
        self._module_absolute_name = None
        super().__init__(*args, **kwargs)

    @abstractmethod
    def init_state(self, batch_size: int, sequence_length: int) -> Dict[str, mx.array]:
        """Initialize the state for this module.

        Args:
            batch_size: Batch size for the state tensors.
            sequence_length: Maximum sequence length for caches.

        Returns:
            Dictionary mapping state names to MLX arrays.
        """
        raise NotImplementedError

    def increment_step(self, state: Dict[str, mx.array], increment: int = 1) -> None:
        """Update state to advance by the given number of steps.

        Args:
            state: The module's state dictionary.
            increment: Number of steps to increment by.
        """
        pass

    def get_state(
        self, model_state: Dict[str, Dict[str, mx.array]]
    ) -> Dict[str, mx.array]:
        """Get the state for this module from the model state.

        Args:
            model_state: Full model state dictionary.

        Returns:
            This module's state dictionary.
        """
        return model_state[self._module_absolute_name]
