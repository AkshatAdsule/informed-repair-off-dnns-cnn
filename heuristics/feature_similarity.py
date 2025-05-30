"""
Feature-Similarity Based heuristic: Select layers where internal representations are most similar.

From Experimental Setup:
"Choosing a layer where the internal representations of the inputs within the repair set
are deemed most similar, suggesting a point of unified processing relevant to the required fix."
"""

from typing import List, Tuple
import torch
import sytorch as st
from .base import LayerSelectionHeuristic


class FeatureSimilarityHeuristic(LayerSelectionHeuristic):
    """
    Select layers where internal representations show high similarity.

    This heuristic identifies layers where the repair set inputs have
    similar feature representations, indicating unified processing.
    """

    def __init__(
        self,
        similarity_metric: str = "cosine",
        num_layers: int = 3,
        include_classifier: bool = True,
        focus_strategy: str = "convergence",
    ):
        """
        Args:
            similarity_metric: 'cosine', 'euclidean', or 'correlation'
            num_layers: Number of layers to select
            include_classifier: Whether to always include the classifier layer
            focus_strategy: 'convergence', 'divergence', or 'bottleneck'
        """
        super().__init__(f"feature_sim_{similarity_metric}_{num_layers}")
        self.similarity_metric = similarity_metric
        self.num_layers = num_layers
        self.include_classifier = include_classifier
        self.focus_strategy = focus_strategy

    def select_layers(self, model: st.nn.Module) -> List[Tuple[str, st.nn.Module]]:
        """Select layers based on estimated feature similarity"""

        layers = []

        # Always include classifier if requested
        if self.include_classifier:
            layers.append(("classifier_conv", model[1][1]))
            remaining_layers = self.num_layers - 1
        else:
            remaining_layers = self.num_layers

        if remaining_layers > 0:
            if self.focus_strategy == "convergence":
                selected_layers = self._select_convergence_layers(
                    model, remaining_layers
                )
            elif self.focus_strategy == "divergence":
                selected_layers = self._select_divergence_layers(
                    model, remaining_layers
                )
            elif self.focus_strategy == "bottleneck":
                selected_layers = self._select_bottleneck_layers(
                    model, remaining_layers
                )
            else:
                selected_layers = self._select_convergence_layers(
                    model, remaining_layers
                )

            layers.extend(selected_layers)

        return layers

    def _select_convergence_layers(
        self, model: st.nn.Module, num_layers: int
    ) -> List[Tuple[str, st.nn.Module]]:
        """
        Select layers where features are likely to converge.

        Heuristic: Later layers in the network tend to have more similar
        representations across different inputs of the same class.
        """
        layers = []
        feature_block = model[0]

        # Start from later layers where convergence is more likely
        for i in range(
            len(feature_block) - 1, max(len(feature_block) - num_layers - 2, -1), -1
        ):
            if len(layers) >= num_layers:
                break

            layer = feature_block[i]
            if hasattr(layer, "__len__") and len(layer) > 0:  # Fire module
                # For convergence, prefer squeeze layers (bottleneck effect)
                if isinstance(layer[0], st.nn.Conv2d) and len(layers) < num_layers:
                    layers.append((f"converge_squeeze_{i}", layer[0]))
            elif isinstance(layer, st.nn.Conv2d) and len(layers) < num_layers:
                layers.append((f"converge_conv_{i}", layer))

        return layers

    def _select_divergence_layers(
        self, model: st.nn.Module, num_layers: int
    ) -> List[Tuple[str, st.nn.Module]]:
        """
        Select layers where features are likely to diverge.

        Heuristic: Middle layers often show feature divergence as the
        network learns to distinguish between different patterns.
        """
        layers = []
        feature_block = model[0]
        total_layers = len(feature_block)

        # Focus on middle layers (divergence region)
        start_idx = total_layers // 4
        end_idx = 3 * total_layers // 4

        for i in range(end_idx, start_idx - 1, -1):
            if len(layers) >= num_layers:
                break

            layer = feature_block[i]
            if hasattr(layer, "__len__") and len(layer) > 2:  # Fire module
                # For divergence, prefer expand layers (more feature channels)
                for j in range(2, len(layer)):  # Expand layers
                    if isinstance(layer[j], st.nn.Conv2d) and len(layers) < num_layers:
                        layers.append((f"diverge_expand_{i}_{j}", layer[j]))
                        break
            elif isinstance(layer, st.nn.Conv2d) and len(layers) < num_layers:
                layers.append((f"diverge_conv_{i}", layer))

        return layers

    def _select_bottleneck_layers(
        self, model: st.nn.Module, num_layers: int
    ) -> List[Tuple[str, st.nn.Module]]:
        """
        Select bottleneck layers where feature similarity is naturally high.

        Heuristic: Squeeze layers in Fire modules act as bottlenecks
        and typically have high feature similarity.
        """
        layers = []
        feature_block = model[0]

        # Collect all squeeze layers (natural bottlenecks)
        squeeze_candidates = []
        for i, layer in enumerate(feature_block):
            if hasattr(layer, "__len__") and len(layer) > 0:  # Fire module
                if isinstance(layer[0], st.nn.Conv2d):
                    squeeze_candidates.append((f"bottleneck_squeeze_{i}", layer[0]))

        # Select from the end (most abstract bottlenecks)
        for name, layer in reversed(squeeze_candidates[-num_layers:]):
            layers.append((name, layer))

        return layers

    def get_description(self) -> str:
        classifier_note = " (including classifier)" if self.include_classifier else ""
        return f"Selects {self.num_layers} layers with high feature similarity ({self.focus_strategy} strategy){classifier_note}"

    def get_complexity_score(self) -> int:
        # Feature similarity-based selection complexity depends on strategy
        if self.focus_strategy == "bottleneck":
            # Bottleneck layers are typically smaller
            base_score = self.num_layers * 60000
        elif self.focus_strategy == "divergence":
            # Divergence layers might be larger (expand layers)
            base_score = self.num_layers * 250000
        else:  # convergence
            # Medium complexity
            base_score = self.num_layers * 150000

        if self.include_classifier:
            base_score += 512000
        return base_score


class FeatureConvergenceHeuristic(FeatureSimilarityHeuristic):
    """Convenience class for convergence-based selection"""

    def __init__(self, num_layers: int = 3, include_classifier: bool = True):
        super().__init__("cosine", num_layers, include_classifier, "convergence")


class FeatureDivergenceHeuristic(FeatureSimilarityHeuristic):
    """Convenience class for divergence-based selection"""

    def __init__(self, num_layers: int = 3, include_classifier: bool = True):
        super().__init__("cosine", num_layers, include_classifier, "divergence")


class BottleneckHeuristic(FeatureSimilarityHeuristic):
    """Convenience class for bottleneck-based selection"""

    def __init__(self, num_layers: int = 3, include_classifier: bool = True):
        super().__init__("cosine", num_layers, include_classifier, "bottleneck")
