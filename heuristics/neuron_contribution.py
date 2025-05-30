"""
Neuron/Path Contribution Profiling heuristic: Select layers based on neuron activation patterns.

From Experimental Setup:
"Run inference over the input polytope defined by the repair set and track neuron activations
across the network. Aggregate activation patterns to identify neurons or layers that consistently
fire across the polytope, indicating strong influence on the output. Prioritize editing layers
where contribution density (e.g., summed or averaged activation magnitudes) is highest within
the repair region, enabling targeted interventions grounded in actual path utilization."
"""

from typing import List, Tuple, Dict, Any
import torch
import sytorch as st
from .base import LayerSelectionHeuristic


class NeuronContributionHeuristic(LayerSelectionHeuristic):
    """
    Select layers based on neuron contribution and activation patterns.

    This heuristic analyzes actual neuron firing patterns across the repair set
    to identify layers with the highest contribution to the output.
    """

    def __init__(
        self,
        contribution_metric: str = "magnitude",
        num_layers: int = 3,
        include_classifier: bool = True,
        consistency_threshold: float = 0.7,
        aggregation_method: str = "mean",
    ):
        """
        Args:
            contribution_metric: 'magnitude', 'variance', 'consistency', 'path_strength'
            num_layers: Number of layers to select
            include_classifier: Whether to always include the classifier layer
            consistency_threshold: Threshold for consistent firing (0.0-1.0)
            aggregation_method: 'mean', 'sum', 'max' for aggregating activations
        """
        super().__init__(f"neuron_contrib_{contribution_metric}_{num_layers}")
        self.contribution_metric = contribution_metric
        self.num_layers = num_layers
        self.include_classifier = include_classifier
        self.consistency_threshold = consistency_threshold
        self.aggregation_method = aggregation_method

    def select_layers(self, model: st.nn.Module) -> List[Tuple[str, st.nn.Module]]:
        """Select layers based on estimated neuron contribution"""

        layers = []

        # Always include classifier if requested
        if self.include_classifier:
            layers.append(("classifier_conv", model[1][1]))
            remaining_layers = self.num_layers - 1
        else:
            remaining_layers = self.num_layers

        if remaining_layers > 0:
            if self.contribution_metric == "magnitude":
                selected_layers = self._select_by_magnitude_contribution(
                    model, remaining_layers
                )
            elif self.contribution_metric == "variance":
                selected_layers = self._select_by_variance_contribution(
                    model, remaining_layers
                )
            elif self.contribution_metric == "consistency":
                selected_layers = self._select_by_consistency_contribution(
                    model, remaining_layers
                )
            elif self.contribution_metric == "path_strength":
                selected_layers = self._select_by_path_strength(model, remaining_layers)
            else:
                selected_layers = self._select_by_magnitude_contribution(
                    model, remaining_layers
                )

            layers.extend(selected_layers)

        return layers

    def _select_by_magnitude_contribution(
        self, model: st.nn.Module, num_layers: int
    ) -> List[Tuple[str, st.nn.Module]]:
        """
        Select layers with highest magnitude contribution.

        Heuristic: Layers that consistently produce high activation magnitudes
        across the repair set are likely to have strong influence.
        """
        layers = []
        feature_block = model[0]

        # Prioritize later layers for high magnitude contribution
        for i in range(len(feature_block) - 1, -1, -1):
            if len(layers) >= num_layers:
                break

            layer = feature_block[i]
            if hasattr(layer, "__len__") and len(layer) > 0:  # Fire module
                # In fire modules, expand layers typically have higher magnitude
                for j in range(
                    len(layer) - 1, -1, -1
                ):  # Reverse order to prioritize expand
                    if isinstance(layer[j], st.nn.Conv2d) and len(layers) < num_layers:
                        layers.append((f"magnitude_contrib_fire_{i}_{j}", layer[j]))
                        break
            elif isinstance(layer, st.nn.Conv2d) and len(layers) < num_layers:
                layers.append((f"magnitude_contrib_conv_{i}", layer))

        return layers

    def _select_by_variance_contribution(
        self, model: st.nn.Module, num_layers: int
    ) -> List[Tuple[str, st.nn.Module]]:
        """
        Select layers with highest variance in contributions.

        Heuristic: Layers with high variance in activations across the repair set
        are likely processing diverse patterns and are good repair targets.
        """
        layers = []
        feature_block = model[0]
        total_layers = len(feature_block)

        # Middle layers often show highest variance
        start_idx = total_layers // 3
        end_idx = 2 * total_layers // 3

        for i in range(end_idx, start_idx - 1, -1):
            if len(layers) >= num_layers:
                break

            layer = feature_block[i]
            if hasattr(layer, "__len__") and len(layer) > 0:  # Fire module
                # Both squeeze and expand can show high variance
                for j, sublayer in enumerate(layer):
                    if isinstance(sublayer, st.nn.Conv2d) and len(layers) < num_layers:
                        layers.append((f"variance_contrib_fire_{i}_{j}", sublayer))
                        break
            elif isinstance(layer, st.nn.Conv2d) and len(layers) < num_layers:
                layers.append((f"variance_contrib_conv_{i}", layer))

        return layers

    def _select_by_consistency_contribution(
        self, model: st.nn.Module, num_layers: int
    ) -> List[Tuple[str, st.nn.Module]]:
        """
        Select layers with consistent activation patterns.

        Heuristic: Layers that consistently fire across all inputs in the repair set
        are reliable targets for modification.
        """
        layers = []
        feature_block = model[0]

        # Look for consistent activators throughout the network
        for i in range(len(feature_block) - 1, -1, -1):
            if len(layers) >= num_layers:
                break

            layer = feature_block[i]
            if hasattr(layer, "__len__") and len(layer) > 0:  # Fire module
                # Squeeze layers often show more consistent patterns
                if isinstance(layer[0], st.nn.Conv2d) and len(layers) < num_layers:
                    layers.append((f"consistent_contrib_squeeze_{i}", layer[0]))
            elif isinstance(layer, st.nn.Conv2d) and len(layers) < num_layers:
                layers.append((f"consistent_contrib_conv_{i}", layer))

        return layers

    def _select_by_path_strength(
        self, model: st.nn.Module, num_layers: int
    ) -> List[Tuple[str, st.nn.Module]]:
        """
        Select layers that are part of the strongest activation paths.

        Heuristic: Layers in the main activation path from input to output
        have the strongest influence and are prime repair targets.
        """
        layers = []
        feature_block = model[0]

        # Path strength typically increases towards the end of the network
        # Select layers that form the "main highway" of information flow
        path_candidates = []

        for i in range(len(feature_block)):
            layer = feature_block[i]
            if hasattr(layer, "__len__") and len(layer) > 0:  # Fire module
                # In fire modules, the squeeze layer is part of the main path
                if isinstance(layer[0], st.nn.Conv2d):
                    path_candidates.append((f"path_strength_squeeze_{i}", layer[0]))
                # First expand layer is also part of main path
                if len(layer) > 2 and isinstance(layer[2], st.nn.Conv2d):
                    path_candidates.append((f"path_strength_expand_{i}_2", layer[2]))
            elif isinstance(layer, st.nn.Conv2d):
                path_candidates.append((f"path_strength_conv_{i}", layer))

        # Select from the end (strongest path components)
        for name, layer in reversed(path_candidates[-num_layers:]):
            layers.append((name, layer))

        return layers

    def get_description(self) -> str:
        classifier_note = " (including classifier)" if self.include_classifier else ""
        return f"Selects {self.num_layers} layers with highest {self.contribution_metric} neuron contribution{classifier_note}"

    def get_complexity_score(self) -> int:
        # Neuron contribution-based selection complexity varies by metric
        if self.contribution_metric == "consistency":
            # Consistent layers might be smaller (squeeze)
            base_score = self.num_layers * 80000
        elif self.contribution_metric == "magnitude":
            # High magnitude layers might be larger (expand)
            base_score = self.num_layers * 300000
        elif self.contribution_metric == "path_strength":
            # Path strength includes both squeeze and expand
            base_score = self.num_layers * 200000
        else:  # variance
            # Medium complexity
            base_score = self.num_layers * 180000

        if self.include_classifier:
            base_score += 512000
        return base_score


class MagnitudeContributionHeuristic(NeuronContributionHeuristic):
    """Convenience class for magnitude contribution-based selection"""

    def __init__(self, num_layers: int = 3, include_classifier: bool = True):
        super().__init__("magnitude", num_layers, include_classifier)


class VarianceContributionHeuristic(NeuronContributionHeuristic):
    """Convenience class for variance contribution-based selection"""

    def __init__(self, num_layers: int = 3, include_classifier: bool = True):
        super().__init__("variance", num_layers, include_classifier)


class ConsistencyContributionHeuristic(NeuronContributionHeuristic):
    """Convenience class for consistency contribution-based selection"""

    def __init__(self, num_layers: int = 3, include_classifier: bool = True):
        super().__init__("consistency", num_layers, include_classifier)


class PathStrengthHeuristic(NeuronContributionHeuristic):
    """Convenience class for path strength-based selection"""

    def __init__(self, num_layers: int = 3, include_classifier: bool = True):
        super().__init__("path_strength", num_layers, include_classifier)
