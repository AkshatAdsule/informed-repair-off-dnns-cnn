#!/usr/bin/env python3
"""
Raw Repair Heuristic - "Raw-dogging" approach that edits all layers with loose bounds

This heuristic represents a brute-force approach where we let the solver modify
any parameter in any layer with very loose constraints, essentially giving it
maximum freedom to find a solution.
"""

import torch
import sytorch as st
from typing import List, Tuple, Any
from .base import LayerSelectionHeuristic


class RawRepairHeuristic(LayerSelectionHeuristic):
    """
    Raw repair heuristic that edits ALL layers with very loose parameter bounds.

    This represents the "raw-dogging" approach where we give the solver maximum
    freedom to modify parameters across the entire network.
    """

    def __init__(self, param_bound: float = 1000.0, include_classifier: bool = True):
        """
        Initialize the raw repair heuristic.

        Args:
            param_bound: Very loose parameter change bound (default: 1000.0 for "almost infinite")
            include_classifier: Whether to include classifier layers
        """
        self.param_bound = param_bound
        self.include_classifier = include_classifier
        name = f"RawRepair_bound{param_bound}"
        super().__init__(name)

    def get_description(self) -> str:
        return (
            f"Raw repair editing ALL layers with loose bounds (±{self.param_bound}). "
            f"Classifier included: {self.include_classifier}"
        )

    def get_complexity_score(self) -> float:
        """High complexity since we're editing everything"""
        return 1.0

    def select_layers(self, model) -> List[Tuple[str, Any]]:
        """
        Select ALL layers for editing.

        Args:
            model: The neural network model

        Returns:
            List of (layer_name, layer) tuples for ALL layers with trainable parameters
        """
        layers_to_edit = []

        for name, module in model.named_modules():
            # Skip the root module
            if name == "":
                continue

            # Check if this module has trainable parameters that we can actually edit
            has_trainable_params = False

            # Check for weight and bias parameters specifically
            if (
                hasattr(module, "weight")
                and module.weight is not None
                and module.weight.requires_grad
            ):
                has_trainable_params = True
            if (
                hasattr(module, "bias")
                and module.bias is not None
                and module.bias.requires_grad
            ):
                has_trainable_params = True

            # Skip layers without trainable parameters (MaxPool, ReLU, Dropout, etc.)
            if not has_trainable_params:
                continue

            # Include classifier if specified
            if "classifier" in name.lower() or "fc" in name.lower():
                if self.include_classifier:
                    layers_to_edit.append((name, module))
            else:
                # Include all other layers with trainable parameters
                layers_to_edit.append((name, module))

        return layers_to_edit

    def validate_selection(self, selected_layers: List[Tuple[str, Any]]) -> bool:
        """
        Validate that we selected a reasonable number of layers.

        Args:
            selected_layers: List of selected layer tuples

        Returns:
            True if selection is valid
        """
        # For raw repair, any number of layers is valid (even all of them)
        return len(selected_layers) > 0


class ConservativeRawRepairHeuristic(LayerSelectionHeuristic):
    """
    Conservative version of raw repair that edits all layers but with moderate bounds.
    """

    def __init__(self, param_bound: float = 50.0, include_classifier: bool = True):
        """
        Initialize the conservative raw repair heuristic.

        Args:
            param_bound: Moderate parameter change bound (default: 50.0)
            include_classifier: Whether to include classifier layers
        """
        self.param_bound = param_bound
        self.include_classifier = include_classifier
        name = f"ConservativeRawRepair_bound{param_bound}"
        super().__init__(name)

    def get_description(self) -> str:
        return (
            f"Conservative raw repair editing ALL layers with moderate bounds (±{self.param_bound}). "
            f"Classifier included: {self.include_classifier}"
        )

    def get_complexity_score(self) -> float:
        """High complexity since we're editing everything"""
        return 0.9

    def select_layers(self, model) -> List[Tuple[str, Any]]:
        """Select ALL layers for editing (same as RawRepairHeuristic)"""
        layers_to_edit = []

        for name, module in model.named_modules():
            # Skip the root module
            if name == "":
                continue

            # Check if this module has trainable parameters that we can actually edit
            has_trainable_params = False

            # Check for weight and bias parameters specifically
            if (
                hasattr(module, "weight")
                and module.weight is not None
                and module.weight.requires_grad
            ):
                has_trainable_params = True
            if (
                hasattr(module, "bias")
                and module.bias is not None
                and module.bias.requires_grad
            ):
                has_trainable_params = True

            # Skip layers without trainable parameters (MaxPool, ReLU, Dropout, etc.)
            if not has_trainable_params:
                continue

            # Include classifier if specified
            if "classifier" in name.lower() or "fc" in name.lower():
                if self.include_classifier:
                    layers_to_edit.append((name, module))
            else:
                # Include all other layers with trainable parameters
                layers_to_edit.append((name, module))

        return layers_to_edit

    def validate_selection(self, selected_layers: List[Tuple[str, Any]]) -> bool:
        """Validate that we selected layers"""
        return len(selected_layers) > 0


class LayerSubsetRawRepairHeuristic(LayerSelectionHeuristic):
    """
    Raw repair that edits all layers but excludes certain layer types.
    """

    def __init__(self, param_bound: float = 100.0, exclude_types: List[str] = None):
        """
        Initialize the layer subset raw repair heuristic.

        Args:
            param_bound: Parameter change bound
            exclude_types: List of layer types to exclude (e.g., ['BatchNorm', 'Dropout'])
        """
        self.param_bound = param_bound
        self.exclude_types = exclude_types or []
        name = f"SubsetRawRepair_bound{param_bound}_exclude{len(self.exclude_types)}"
        super().__init__(name)

    def get_description(self) -> str:
        excluded = ", ".join(self.exclude_types) if self.exclude_types else "none"
        return (
            f"Raw repair editing most layers (±{self.param_bound}), "
            f"excluding: {excluded}"
        )

    def get_complexity_score(self) -> float:
        """Complexity depends on how much we exclude"""
        return max(0.1, 1.0 - 0.1 * len(self.exclude_types))

    def select_layers(self, model) -> List[Tuple[str, Any]]:
        """Select layers excluding certain types"""
        layers_to_edit = []

        for name, module in model.named_modules():
            # Skip the root module
            if name == "":
                continue

            # Check if this module has trainable parameters that we can actually edit
            has_trainable_params = False

            # Check for weight and bias parameters specifically
            if (
                hasattr(module, "weight")
                and module.weight is not None
                and module.weight.requires_grad
            ):
                has_trainable_params = True
            if (
                hasattr(module, "bias")
                and module.bias is not None
                and module.bias.requires_grad
            ):
                has_trainable_params = True

            # Skip layers without trainable parameters (MaxPool, ReLU, Dropout, etc.)
            if not has_trainable_params:
                continue

            # Check if we should exclude this layer type
            module_type = module.__class__.__name__
            if any(exclude_type in module_type for exclude_type in self.exclude_types):
                continue

            layers_to_edit.append((name, module))

        return layers_to_edit

    def validate_selection(self, selected_layers: List[Tuple[str, Any]]) -> bool:
        """Validate that we selected layers"""
        return len(selected_layers) > 0
