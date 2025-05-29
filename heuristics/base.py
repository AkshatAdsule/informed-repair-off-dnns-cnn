"""
Base class for layer selection heuristics in neural network repair.
"""

from typing import List, Tuple
import sytorch as st
from abc import ABC, abstractmethod


class LayerSelectionHeuristic(ABC):
    """Base class for layer selection heuristics"""

    def __init__(self, name: str):
        """
        Initialize the heuristic with a descriptive name.

        Args:
            name: A unique name identifying this heuristic
        """
        self.name = name

    @abstractmethod
    def select_layers(self, model: st.nn.Module) -> List[Tuple[str, st.nn.Module]]:
        """
        Select layers to make symbolic for repair.

        Args:
            model: The SyTorch model to analyze

        Returns:
            List of (layer_path, layer) tuples to make symbolic
        """
        raise NotImplementedError

    def get_description(self) -> str:
        """
        Get a human-readable description of this heuristic.
        Override this to provide detailed descriptions.
        """
        return f"Heuristic: {self.name}"

    def get_complexity_score(self) -> int:
        """
        Get a rough complexity score (number of parameters that might be modified).
        Override this to provide better estimates for your heuristic.

        Returns:
            Estimated number of parameters this heuristic might modify
        """
        return 1  # Default conservative estimate

    def validate_selection(self, layers: List[Tuple[str, st.nn.Module]]) -> bool:
        """
        Validate that the selected layers are appropriate for repair.
        Override this to add custom validation logic.

        Args:
            layers: The selected layers

        Returns:
            True if selection is valid, False otherwise
        """
        # Basic validation: at least one layer selected
        if not layers:
            return False

        # Check that all selected items are actually layers
        for name, layer in layers:
            if not isinstance(layer, st.nn.Module):
                return False

        return True

    def __str__(self) -> str:
        return self.get_description()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
