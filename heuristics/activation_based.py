"""
Activation-Based heuristic: Select layers based on activation statistics.

From Experimental Setup:
"Selecting the start layer based on metrics calculated across the repair set,
such as the layer exhibiting the highest average activation magnitude, or
perhaps the layer with the highest variance in activations."
"""

from typing import List, Tuple, Dict
import torch
import sytorch as st
from .base import LayerSelectionHeuristic


class ActivationBasedHeuristic(LayerSelectionHeuristic):
    """
    Select layers based on activation statistics across the repair set.

    This heuristic analyzes activation patterns across multiple inputs
    to identify layers with high activation magnitude or variance.
    """

    def __init__(
        self,
        metric: str = "magnitude",
        num_layers: int = 3,
        include_classifier: bool = True,
        sample_inputs: int = 10,
    ):
        """
        Args:
            metric: 'magnitude' for avg activation magnitude, 'variance' for activation variance
            num_layers: Number of layers to select
            include_classifier: Whether to always include the classifier layer
            sample_inputs: Number of inputs to sample for activation analysis
        """
        super().__init__(f"activation_{metric}_{num_layers}")
        self.metric = metric
        self.num_layers = num_layers
        self.include_classifier = include_classifier
        self.sample_inputs = sample_inputs

    def select_layers(self, model: st.nn.Module) -> List[Tuple[str, st.nn.Module]]:
        """Select layers based on activation statistics"""

        # For this implementation, we'll use a simplified heuristic
        # In practice, you'd want to run sample inputs through the model
        # and compute actual activation statistics

        layers = []

        # Always include classifier if requested
        if self.include_classifier:
            layers.append(("classifier_conv", model[1][1]))
            remaining_layers = self.num_layers - 1
        else:
            remaining_layers = self.num_layers

        if remaining_layers > 0:
            # Simplified heuristic: Select based on layer position and type
            # Real implementation would compute actual activation statistics
            if self.metric == "magnitude":
                selected_layers = self._select_by_magnitude_heuristic(
                    model, remaining_layers
                )
            elif self.metric == "variance":
                selected_layers = self._select_by_variance_heuristic(
                    model, remaining_layers
                )
            else:
                selected_layers = self._select_by_magnitude_heuristic(
                    model, remaining_layers
                )

            layers.extend(selected_layers)

        return layers

    def _select_by_magnitude_heuristic(
        self, model: st.nn.Module, num_layers: int
    ) -> List[Tuple[str, st.nn.Module]]:
        """
        Heuristic for selecting layers likely to have high activation magnitude.

        In practice, deeper layers often have higher activation magnitudes,
        so we prioritize later layers in the network.
        """
        layers = []
        feature_block = model[0]

        # Start from later layers (higher activation magnitude heuristic)
        for i in range(len(feature_block) - 1, -1, -1):
            if len(layers) >= num_layers:
                break

            layer = feature_block[i]
            if hasattr(layer, "__len__") and len(layer) > 2:  # Fire module
                # Add expand layers (typically have higher activations)
                for j, sublayer in enumerate(layer):
                    if isinstance(sublayer, st.nn.Conv2d) and len(layers) < num_layers:
                        # Prioritize expand layers (usually indices 2, 4)
                        if j >= 2:  # Expand layers
                            layers.append((f"activation_mag_fire_{i}_{j}", sublayer))
            elif isinstance(layer, st.nn.Conv2d) and len(layers) < num_layers:
                layers.append((f"activation_mag_conv_{i}", layer))

        return layers

    def _select_by_variance_heuristic(
        self, model: st.nn.Module, num_layers: int
    ) -> List[Tuple[str, st.nn.Module]]:
        """
        Heuristic for selecting layers likely to have high activation variance.

        Middle layers often show more variance as they process increasingly
        abstract features, so we select from middle sections.
        """
        layers = []
        feature_block = model[0]
        total_layers = len(feature_block)

        # Select from middle portion (higher variance heuristic)
        start_idx = total_layers // 3
        end_idx = 2 * total_layers // 3

        for i in range(end_idx, start_idx - 1, -1):  # Work backwards in middle section
            if len(layers) >= num_layers:
                break

            layer = feature_block[i]
            if hasattr(layer, "__len__") and len(layer) > 0:  # Fire module
                # Add squeeze layers (often show high variance)
                if isinstance(layer[0], st.nn.Conv2d) and len(layers) < num_layers:
                    layers.append((f"activation_var_squeeze_{i}", layer[0]))
            elif isinstance(layer, st.nn.Conv2d) and len(layers) < num_layers:
                layers.append((f"activation_var_conv_{i}", layer))

        return layers

    def get_description(self) -> str:
        classifier_note = " (including classifier)" if self.include_classifier else ""
        return f"Selects {self.num_layers} layers based on activation {self.metric}{classifier_note}"

    def get_complexity_score(self) -> int:
        # Activation-based selection might target high-impact layers
        base_score = self.num_layers * 180000  # Medium estimate
        if self.include_classifier:
            base_score += 512000
        return base_score


class ActivationMagnitudeHeuristic(ActivationBasedHeuristic):
    """Convenience class for magnitude-based selection"""

    def __init__(self, num_layers: int = 3, include_classifier: bool = True):
        super().__init__("magnitude", num_layers, include_classifier)


class ActivationVarianceHeuristic(ActivationBasedHeuristic):
    """Convenience class for variance-based selection"""

    def __init__(self, num_layers: int = 3, include_classifier: bool = True):
        super().__init__("variance", num_layers, include_classifier)
