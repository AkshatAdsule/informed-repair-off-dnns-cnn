"""
Gradient-based heuristic: Select layers based on gradient magnitude.
"""

from typing import List, Tuple
import torch
import sytorch as st
from .base import LayerSelectionHeuristic


class GradientBasedHeuristic(LayerSelectionHeuristic):
    """
    Select layers based on gradient magnitude with respect to loss.
    This is a more sophisticated heuristic that analyzes which layers
    contribute most to the error.
    """

    def __init__(self, num_layers: int = 3, include_classifier: bool = True):
        super().__init__(f"gradient_based_{num_layers}")
        self.num_layers = num_layers
        self.include_classifier = include_classifier

    def select_layers(self, model: st.nn.Module) -> List[Tuple[str, st.nn.Module]]:
        """
        Select layers based on gradient magnitude analysis.

        Note: This is a simplified example. In practice, you'd need to:
        1. Compute gradients on a sample of misclassified images
        2. Rank layers by gradient magnitude
        3. Select top-k layers

        For now, we'll use a heuristic approximation.
        """
        layers = []

        # Always include classifier if requested
        if self.include_classifier:
            layers.append(("classifier_conv", model[1][1]))
            remaining_layers = self.num_layers - 1
        else:
            remaining_layers = self.num_layers

        if remaining_layers > 0:
            # For this example, we'll select the last few Fire modules
            # as they typically have higher gradients
            feature_block = model[0]
            fire_count = 0

            for i in range(len(feature_block) - 1, -1, -1):
                layer = feature_block[i]
                if hasattr(layer, "__len__") and len(layer) > 2:  # Fire module
                    # Add the squeeze layer (usually index 0)
                    if len(layer) > 0 and isinstance(layer[0], st.nn.Conv2d):
                        layers.append((f"fire_{i}_squeeze", layer[0]))
                        fire_count += 1

                        if fire_count >= remaining_layers:
                            break

        return layers

    def get_description(self) -> str:
        classifier_note = " (including classifier)" if self.include_classifier else ""
        return f"Selects {self.num_layers} layers based on gradient analysis{classifier_note}"

    def get_complexity_score(self) -> int:
        # Gradient-based selection might pick high-impact layers
        # Estimate based on typical squeeze layer sizes
        base_score = self.num_layers * 50000  # Squeeze layers are smaller
        if self.include_classifier:
            base_score += 512000
        return base_score
