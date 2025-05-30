"""
Layer Type/Position heuristic: Simple structural heuristics based on layer type and position.

From Experimental Setup:
"Simple heuristics like always choosing the first fully-connected layer after convolutional blocks,
or the penultimate layer."
"""

from typing import List, Tuple
import torch
import sytorch as st
from .base import LayerSelectionHeuristic


class LayerPositionHeuristic(LayerSelectionHeuristic):
    """
    Select layers based on structural position and type.

    This implements simple, architecture-aware heuristics that
    select layers based on their position and type in the network.
    """

    def __init__(
        self,
        position_strategy: str = "penultimate",
        num_layers: int = 3,
        layer_type_preference: str = "conv",
    ):
        """
        Args:
            position_strategy: 'penultimate', 'first_fc', 'last_conv', 'transition', 'early', 'late'
            num_layers: Number of layers to select
            layer_type_preference: 'conv', 'squeeze', 'expand', 'mixed'
        """
        super().__init__(f"position_{position_strategy}_{num_layers}")
        self.position_strategy = position_strategy
        self.num_layers = num_layers
        self.layer_type_preference = layer_type_preference

    def select_layers(self, model: st.nn.Module) -> List[Tuple[str, st.nn.Module]]:
        """Select layers based on position strategy"""

        if self.position_strategy == "penultimate":
            return self._select_penultimate_layers(model)
        elif self.position_strategy == "first_fc":
            return self._select_first_fc_layer(model)
        elif self.position_strategy == "last_conv":
            return self._select_last_conv_layers(model)
        elif self.position_strategy == "transition":
            return self._select_transition_layers(model)
        elif self.position_strategy == "early":
            return self._select_early_layers(model)
        elif self.position_strategy == "late":
            return self._select_late_layers(model)
        else:
            return self._select_penultimate_layers(model)

    def _select_penultimate_layers(
        self, model: st.nn.Module
    ) -> List[Tuple[str, st.nn.Module]]:
        """
        Select the penultimate layer(s) - layers just before the final classifier.

        For SqueezeNet, this would be the last few Fire modules.
        """
        layers = []

        # Always include classifier (the ultimate layer)
        layers.append(("classifier_conv", model[1][1]))

        if self.num_layers > 1:
            feature_block = model[0]
            remaining = self.num_layers - 1

            # Get penultimate layers (last few from features)
            for i in range(len(feature_block) - 1, -1, -1):
                if (
                    len(layers) - 1 >= remaining
                ):  # -1 because classifier is already added
                    break

                layer = feature_block[i]
                if hasattr(layer, "__len__") and len(layer) > 0:  # Fire module
                    if self.layer_type_preference == "squeeze":
                        if isinstance(layer[0], st.nn.Conv2d):
                            layers.append((f"penult_squeeze_{i}", layer[0]))
                    elif self.layer_type_preference == "expand":
                        for j in range(2, len(layer)):  # Expand layers
                            if isinstance(layer[j], st.nn.Conv2d):
                                layers.append((f"penult_expand_{i}_{j}", layer[j]))
                                break
                    else:  # mixed or conv
                        for j, sublayer in enumerate(layer):
                            if isinstance(sublayer, st.nn.Conv2d):
                                layers.append((f"penult_fire_{i}_{j}", sublayer))
                                break
                elif isinstance(layer, st.nn.Conv2d):
                    layers.append((f"penult_conv_{i}", layer))

        return layers

    def _select_first_fc_layer(
        self, model: st.nn.Module
    ) -> List[Tuple[str, st.nn.Module]]:
        """
        Select the first fully-connected layer after convolutional blocks.

        For SqueezeNet, the classifier conv layer serves as the "first FC" layer.
        """
        layers = []

        # In SqueezeNet, the classifier conv is the first FC-like layer
        layers.append(("first_fc_classifier", model[1][1]))

        # If we need more layers, add the last few feature layers
        if self.num_layers > 1:
            feature_block = model[0]
            remaining = self.num_layers - 1

            for i in range(
                len(feature_block) - 1, max(len(feature_block) - remaining - 1, -1), -1
            ):
                layer = feature_block[i]
                if hasattr(layer, "__len__") and len(layer) > 0:  # Fire module
                    # Add the most FC-like layer from fire module (squeeze)
                    if isinstance(layer[0], st.nn.Conv2d):
                        layers.append((f"fc_like_squeeze_{i}", layer[0]))
                elif isinstance(layer, st.nn.Conv2d):
                    layers.append((f"fc_like_conv_{i}", layer))

        return layers

    def _select_last_conv_layers(
        self, model: st.nn.Module
    ) -> List[Tuple[str, st.nn.Module]]:
        """
        Select the last few convolutional layers before classification.
        """
        layers = []
        feature_block = model[0]

        # Get last conv layers from features (before classifier)
        for i in range(len(feature_block) - 1, -1, -1):
            if len(layers) >= self.num_layers:
                break

            layer = feature_block[i]
            if hasattr(layer, "__len__") and len(layer) > 0:  # Fire module
                for j, sublayer in enumerate(layer):
                    if (
                        isinstance(sublayer, st.nn.Conv2d)
                        and len(layers) < self.num_layers
                    ):
                        layers.append((f"last_conv_fire_{i}_{j}", sublayer))
            elif isinstance(layer, st.nn.Conv2d) and len(layers) < self.num_layers:
                layers.append((f"last_conv_{i}", layer))

        return layers

    def _select_transition_layers(
        self, model: st.nn.Module
    ) -> List[Tuple[str, st.nn.Module]]:
        """
        Select layers at transition points (e.g., around pooling layers).
        """
        layers = []
        feature_block = model[0]

        # Include classifier as a major transition point
        if len(layers) < self.num_layers:
            layers.append(("transition_classifier", model[1][1]))

        # Look for layers around pooling operations
        # In SqueezeNet, pooling typically happens after certain fire modules
        pooling_positions = []
        for i, layer in enumerate(feature_block):
            if isinstance(layer, st.nn.MaxPool2d):
                pooling_positions.append(i)

        # Select layers around pooling positions
        for pool_pos in reversed(pooling_positions):
            if len(layers) >= self.num_layers:
                break

            # Layer before pooling
            if pool_pos > 0:
                prev_layer = feature_block[pool_pos - 1]
                if (
                    hasattr(prev_layer, "__len__") and len(prev_layer) > 0
                ):  # Fire module
                    if isinstance(prev_layer[0], st.nn.Conv2d):
                        layers.append(
                            (f"transition_pre_pool_{pool_pos}", prev_layer[0])
                        )
                elif isinstance(prev_layer, st.nn.Conv2d):
                    layers.append((f"transition_pre_pool_{pool_pos}", prev_layer))

        return layers

    def _select_early_layers(
        self, model: st.nn.Module
    ) -> List[Tuple[str, st.nn.Module]]:
        """Select early layers in the network."""
        layers = []
        feature_block = model[0]

        # Select from first few layers
        for i in range(min(self.num_layers * 2, len(feature_block))):
            if len(layers) >= self.num_layers:
                break

            layer = feature_block[i]
            if hasattr(layer, "__len__") and len(layer) > 0:  # Fire module
                if isinstance(layer[0], st.nn.Conv2d):
                    layers.append((f"early_squeeze_{i}", layer[0]))
            elif isinstance(layer, st.nn.Conv2d):
                layers.append((f"early_conv_{i}", layer))

        return layers

    def _select_late_layers(
        self, model: st.nn.Module
    ) -> List[Tuple[str, st.nn.Module]]:
        """Select late layers in the network."""
        layers = []

        # Include classifier
        layers.append(("late_classifier", model[1][1]))

        if self.num_layers > 1:
            feature_block = model[0]
            remaining = self.num_layers - 1

            # Select from last few feature layers
            for i in range(
                len(feature_block) - 1, max(len(feature_block) - remaining - 2, -1), -1
            ):
                layer = feature_block[i]
                if hasattr(layer, "__len__") and len(layer) > 0:  # Fire module
                    if isinstance(layer[0], st.nn.Conv2d):
                        layers.append((f"late_squeeze_{i}", layer[0]))
                elif isinstance(layer, st.nn.Conv2d):
                    layers.append((f"late_conv_{i}", layer))

        return layers

    def get_description(self) -> str:
        return f"Selects {self.num_layers} layers using {self.position_strategy} position strategy (prefers {self.layer_type_preference} layers)"

    def get_complexity_score(self) -> int:
        # Position-based complexity depends on strategy
        if self.position_strategy in ["first_fc", "penultimate"]:
            # These typically select high-parameter layers
            base_score = self.num_layers * 400000
        elif self.position_strategy == "early":
            # Early layers are typically smaller
            base_score = self.num_layers * 100000
        else:
            # Medium complexity for other strategies
            base_score = self.num_layers * 200000

        return base_score


class PenultimateLayerHeuristic(LayerPositionHeuristic):
    """Convenience class for penultimate layer selection"""

    def __init__(self, num_layers: int = 2):
        super().__init__("penultimate", num_layers, "mixed")


class FirstFCHeuristic(LayerPositionHeuristic):
    """Convenience class for first FC layer selection"""

    def __init__(self, num_layers: int = 1):
        super().__init__("first_fc", num_layers, "conv")


class TransitionLayerHeuristic(LayerPositionHeuristic):
    """Convenience class for transition layer selection"""

    def __init__(self, num_layers: int = 3):
        super().__init__("transition", num_layers, "squeeze")
