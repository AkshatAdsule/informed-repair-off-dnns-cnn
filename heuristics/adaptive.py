"""
Adaptive heuristic: Dynamically select layers based on model analysis.
"""

from typing import List, Tuple, Optional
import torch
import sytorch as st
from .base import LayerSelectionHeuristic


class AdaptiveHeuristic(LayerSelectionHeuristic):
    """
    Adaptive layer selection that changes strategy based on model characteristics
    and error patterns.
    """

    def __init__(
        self,
        strategy: str = "confidence_based",
        max_layers: int = 4,
        min_layers: int = 1,
    ):
        super().__init__(f"adaptive_{strategy}_{max_layers}")
        self.strategy = strategy
        self.max_layers = max_layers
        self.min_layers = min_layers

    def select_layers(self, model: st.nn.Module) -> List[Tuple[str, st.nn.Module]]:
        """Select layers adaptively based on the chosen strategy"""

        if self.strategy == "confidence_based":
            return self._confidence_based_selection(model)
        elif self.strategy == "architecture_aware":
            return self._architecture_aware_selection(model)
        elif self.strategy == "layer_size_balanced":
            return self._layer_size_balanced_selection(model)
        else:
            # Fallback to simple heuristic
            return self._simple_fallback(model)

    def _confidence_based_selection(
        self, model: st.nn.Module
    ) -> List[Tuple[str, st.nn.Module]]:
        """
        Select layers based on predicted confidence.
        Lower confidence might indicate need for more extensive repair.
        """
        layers = []

        # Always start with classifier
        layers.append(("classifier_conv", model[1][1]))

        # For this example, assume we need more layers for complex repairs
        # In practice, you'd analyze prediction confidence
        num_additional = min(self.max_layers - 1, 2)  # Conservative approach

        if num_additional > 0:
            feature_block = model[0]
            added = 0

            # Add recent Fire modules
            for i in range(len(feature_block) - 1, -1, -1):
                layer = feature_block[i]
                if hasattr(layer, "__len__") and len(layer) > 2:
                    # Add squeeze and expand layers
                    for j, sublayer in enumerate(layer):
                        if (
                            isinstance(sublayer, st.nn.Conv2d)
                            and added < num_additional
                        ):
                            layers.append((f"adaptive_fire_{i}_{j}", sublayer))
                            added += 1

                    if added >= num_additional:
                        break

        return layers

    def _architecture_aware_selection(
        self, model: st.nn.Module
    ) -> List[Tuple[str, st.nn.Module]]:
        """Select layers respecting architectural boundaries"""
        layers = []

        # Classifier
        layers.append(("classifier_conv", model[1][1]))

        # Select complete Fire modules rather than individual layers
        feature_block = model[0]
        modules_to_add = min(
            (self.max_layers - 1) // 3, 2
        )  # Each module has ~3 conv layers

        module_count = 0
        for i in range(len(feature_block) - 1, -1, -1):
            layer = feature_block[i]
            if (
                hasattr(layer, "__len__")
                and len(layer) > 2
                and module_count < modules_to_add
            ):
                # Add all conv layers in this Fire module
                for j, sublayer in enumerate(layer):
                    if isinstance(sublayer, st.nn.Conv2d):
                        layers.append((f"arch_aware_fire_{i}_{j}", sublayer))
                module_count += 1

        return layers

    def _layer_size_balanced_selection(
        self, model: st.nn.Module
    ) -> List[Tuple[str, st.nn.Module]]:
        """Select layers to balance parameter count vs. impact"""
        layers = []

        # Start with classifier (large impact, medium size)
        layers.append(("classifier_conv", model[1][1]))

        # Add smaller layers that might have high impact
        feature_block = model[0]
        remaining_budget = self.max_layers - 1

        # Prefer squeeze layers (smaller but potentially high impact)
        for i in range(len(feature_block) - 1, -1, -1):
            if remaining_budget <= 0:
                break

            layer = feature_block[i]
            if hasattr(layer, "__len__") and len(layer) > 0:
                # Add squeeze layer (typically smaller)
                if isinstance(layer[0], st.nn.Conv2d):
                    layers.append((f"balanced_squeeze_{i}", layer[0]))
                    remaining_budget -= 1

        return layers

    def _simple_fallback(self, model: st.nn.Module) -> List[Tuple[str, st.nn.Module]]:
        """Simple fallback strategy"""
        return [("classifier_conv", model[1][1])]

    def get_description(self) -> str:
        return f"Adaptive selection using {self.strategy} strategy (max {self.max_layers} layers)"

    def get_complexity_score(self) -> int:
        # Adaptive strategies might vary widely in complexity
        if self.strategy == "confidence_based":
            return self.max_layers * 150000  # Medium estimate
        elif self.strategy == "architecture_aware":
            return self.max_layers * 200000  # Higher for complete modules
        elif self.strategy == "layer_size_balanced":
            return self.max_layers * 100000  # Lower for size-conscious selection
        else:
            return 512000  # Fallback estimate
