"""
Gradient/Sensitivity-Based heuristic: Select layers with highest parameter sensitivity.

From Experimental Setup:
"Identifying layers where parameters show the most sensitivity (e.g., largest gradient norms)
with respect to the inputs in the repair set. This might indicate layers most influential
on the incorrect output."
"""

from typing import List, Tuple
import torch
import sytorch as st
from .base import LayerSelectionHeuristic


class SensitivityBasedHeuristic(LayerSelectionHeuristic):
    """
    Select layers based on parameter sensitivity (gradient norms).

    This heuristic identifies layers where parameters show the most
    sensitivity to changes, indicating high influence on outputs.
    """

    def __init__(
        self,
        sensitivity_metric: str = "gradient_norm",
        num_layers: int = 3,
        include_classifier: bool = True,
        layer_focus: str = "all",
    ):
        """
        Args:
            sensitivity_metric: 'gradient_norm', 'parameter_change', or 'output_sensitivity'
            num_layers: Number of layers to select
            include_classifier: Whether to always include the classifier layer
            layer_focus: 'all', 'conv_only', 'recent' for layer selection focus
        """
        super().__init__(f"sensitivity_{sensitivity_metric}_{num_layers}")
        self.sensitivity_metric = sensitivity_metric
        self.num_layers = num_layers
        self.include_classifier = include_classifier
        self.layer_focus = layer_focus

    def select_layers(self, model: st.nn.Module) -> List[Tuple[str, st.nn.Module]]:
        """Select layers based on estimated sensitivity"""

        layers = []

        # Always include classifier if requested
        if self.include_classifier:
            layers.append(("classifier_conv", model[1][1]))
            remaining_layers = self.num_layers - 1
        else:
            remaining_layers = self.num_layers

        if remaining_layers > 0:
            if self.sensitivity_metric == "gradient_norm":
                selected_layers = self._select_by_gradient_sensitivity(
                    model, remaining_layers
                )
            elif self.sensitivity_metric == "parameter_change":
                selected_layers = self._select_by_parameter_sensitivity(
                    model, remaining_layers
                )
            elif self.sensitivity_metric == "output_sensitivity":
                selected_layers = self._select_by_output_sensitivity(
                    model, remaining_layers
                )
            else:
                selected_layers = self._select_by_gradient_sensitivity(
                    model, remaining_layers
                )

            layers.extend(selected_layers)

        return layers

    def _select_by_gradient_sensitivity(
        self, model: st.nn.Module, num_layers: int
    ) -> List[Tuple[str, st.nn.Module]]:
        """
        Select layers likely to have high gradient norms.

        Heuristic: Later layers typically have higher gradient magnitudes
        due to closer proximity to the loss function.
        """
        layers = []
        feature_block = model[0]

        if self.layer_focus == "recent":
            # Focus on recent layers (high gradient sensitivity)
            search_range = range(
                len(feature_block) - 1, max(len(feature_block) - 6, -1), -1
            )
        else:
            # Search all layers, prioritizing later ones
            search_range = range(len(feature_block) - 1, -1, -1)

        for i in search_range:
            if len(layers) >= num_layers:
                break

            layer = feature_block[i]
            if hasattr(layer, "__len__") and len(layer) > 0:  # Fire module
                # Add layers that typically have high gradients
                for j, sublayer in enumerate(layer):
                    if isinstance(sublayer, st.nn.Conv2d) and len(layers) < num_layers:
                        layers.append((f"grad_sens_fire_{i}_{j}", sublayer))
            elif isinstance(layer, st.nn.Conv2d) and len(layers) < num_layers:
                layers.append((f"grad_sens_conv_{i}", layer))

        return layers

    def _select_by_parameter_sensitivity(
        self, model: st.nn.Module, num_layers: int
    ) -> List[Tuple[str, st.nn.Module]]:
        """
        Select layers where parameters are most sensitive to changes.

        Heuristic: Layers with smaller parameter counts often show higher
        relative sensitivity to parameter changes.
        """
        layers = []
        feature_block = model[0]

        # For SqueezeNet, squeeze layers have fewer parameters and higher sensitivity
        for i in range(len(feature_block) - 1, -1, -1):
            if len(layers) >= num_layers:
                break

            layer = feature_block[i]
            if hasattr(layer, "__len__") and len(layer) > 0:  # Fire module
                # Prioritize squeeze layers (smaller, more sensitive)
                if isinstance(layer[0], st.nn.Conv2d) and len(layers) < num_layers:
                    layers.append((f"param_sens_squeeze_{i}", layer[0]))

        return layers

    def _select_by_output_sensitivity(
        self, model: st.nn.Module, num_layers: int
    ) -> List[Tuple[str, st.nn.Module]]:
        """
        Select layers that most directly influence output.

        Heuristic: Later layers have more direct influence on final output.
        """
        layers = []
        feature_block = model[0]

        # Select from last few fire modules (high output sensitivity)
        fire_count = 0
        for i in range(len(feature_block) - 1, -1, -1):
            if len(layers) >= num_layers:
                break

            layer = feature_block[i]
            if hasattr(layer, "__len__") and len(layer) > 2:  # Fire module
                # Add all conv layers from this fire module
                for j, sublayer in enumerate(layer):
                    if isinstance(sublayer, st.nn.Conv2d) and len(layers) < num_layers:
                        layers.append((f"output_sens_fire_{i}_{j}", sublayer))

                fire_count += 1
                if fire_count >= 2:  # Limit to last 2 fire modules
                    break

        return layers

    def get_description(self) -> str:
        classifier_note = " (including classifier)" if self.include_classifier else ""
        return f"Selects {self.num_layers} layers with highest {self.sensitivity_metric} sensitivity{classifier_note}"

    def get_complexity_score(self) -> int:
        # Sensitivity-based selection might target high-impact layers
        if self.sensitivity_metric == "parameter_change":
            # Smaller layers (squeeze) = lower complexity
            base_score = self.num_layers * 80000
        else:
            # Gradient/output sensitive layers might be larger
            base_score = self.num_layers * 220000

        if self.include_classifier:
            base_score += 512000
        return base_score


class GradientNormHeuristic(SensitivityBasedHeuristic):
    """Convenience class for gradient norm-based selection"""

    def __init__(self, num_layers: int = 3, include_classifier: bool = True):
        super().__init__("gradient_norm", num_layers, include_classifier)


class ParameterSensitivityHeuristic(SensitivityBasedHeuristic):
    """Convenience class for parameter sensitivity-based selection"""

    def __init__(self, num_layers: int = 3, include_classifier: bool = True):
        super().__init__("parameter_change", num_layers, include_classifier)


class OutputSensitivityHeuristic(SensitivityBasedHeuristic):
    """Convenience class for output sensitivity-based selection"""

    def __init__(self, num_layers: int = 3, include_classifier: bool = True):
        super().__init__("output_sensitivity", num_layers, include_classifier)
