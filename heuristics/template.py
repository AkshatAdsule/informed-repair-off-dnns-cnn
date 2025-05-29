"""
Template for creating new layer selection heuristics.

Copy this file and modify it to create your own heuristic.
Don't forget to add your new heuristic to __init__.py!
"""

from typing import List, Tuple
import sytorch as st
from .base import LayerSelectionHeuristic


class TemplateHeuristic(LayerSelectionHeuristic):
    """
    Template heuristic - replace this with your description.

    This class shows the minimal interface you need to implement
    to create a new heuristic.
    """

    def __init__(self, your_parameter: int = 1):
        # Create a descriptive name for your heuristic
        super().__init__(f"template_{your_parameter}")

        # Store your parameters
        self.your_parameter = your_parameter

    def select_layers(self, model: st.nn.Module) -> List[Tuple[str, st.nn.Module]]:
        """
        REQUIRED: Implement this method to select layers.

        Args:
            model: The SyTorch model to analyze

        Returns:
            List of (layer_name, layer_object) tuples to make symbolic
        """
        layers = []

        # Example: Always include classifier
        layers.append(("classifier_conv", model[1][1]))

        # TODO: Add your layer selection logic here
        # Some common patterns:

        # 1. Iterate through feature layers:
        # feature_block = model[0]
        # for i, layer in enumerate(feature_block):
        #     if isinstance(layer, st.nn.Conv2d):
        #         layers.append((f"feature_{i}", layer))

        # 2. Select Fire modules:
        # for i in range(len(feature_block) - 1, -1, -1):
        #     layer = feature_block[i]
        #     if hasattr(layer, "__len__") and len(layer) > 2:  # Fire module
        #         for j, sublayer in enumerate(layer):
        #             if isinstance(sublayer, st.nn.Conv2d):
        #                 layers.append((f"fire_{i}_{j}", sublayer))

        # 3. Use your parameter to control selection:
        # if self.your_parameter > 1:
        #     # Add more layers based on parameter value
        #     pass

        return layers

    def get_description(self) -> str:
        """
        OPTIONAL: Provide a human-readable description.
        """
        return f"Template heuristic with parameter {self.your_parameter}"

    def get_complexity_score(self) -> int:
        """
        OPTIONAL: Estimate complexity (number of parameters).

        This helps in analysis and comparison of heuristics.
        """
        # Rough estimate based on your selection strategy
        return self.your_parameter * 100000  # Example calculation

    def validate_selection(self, layers: List[Tuple[str, st.nn.Module]]) -> bool:
        """
        OPTIONAL: Add custom validation for your selection.

        Override this if you have specific requirements.
        """
        # Call parent validation first
        if not super().validate_selection(layers):
            return False

        # Add your custom validation here
        # Example: Ensure we have at least one layer
        return len(layers) >= 1


# Example of a more complex heuristic
class ExampleComplexHeuristic(LayerSelectionHeuristic):
    """
    Example of a more sophisticated heuristic.

    This shows how to implement more complex selection logic.
    """

    def __init__(
        self,
        strategy: str = "depth_based",
        max_layers: int = 3,
        prioritize_recent: bool = True,
    ):
        super().__init__(f"complex_{strategy}_{max_layers}")
        self.strategy = strategy
        self.max_layers = max_layers
        self.prioritize_recent = prioritize_recent

    def select_layers(self, model: st.nn.Module) -> List[Tuple[str, st.nn.Module]]:
        """Complex selection based on multiple criteria"""

        if self.strategy == "depth_based":
            return self._depth_based_selection(model)
        elif self.strategy == "size_based":
            return self._size_based_selection(model)
        else:
            # Fallback
            return [("classifier_conv", model[1][1])]

    def _depth_based_selection(
        self, model: st.nn.Module
    ) -> List[Tuple[str, st.nn.Module]]:
        """Select layers based on depth in network"""
        layers = []

        # Always include classifier
        layers.append(("classifier_conv", model[1][1]))

        # Select from different depths
        feature_block = model[0]
        total_layers = len(feature_block)

        # Select layers at different depth intervals
        for i in range(min(self.max_layers - 1, total_layers)):
            if self.prioritize_recent:
                idx = total_layers - 1 - i  # Work backwards
            else:
                idx = i  # Work forwards

            layer = feature_block[idx]
            if isinstance(layer, st.nn.Conv2d):
                layers.append((f"depth_{idx}", layer))

        return layers

    def _size_based_selection(
        self, model: st.nn.Module
    ) -> List[Tuple[str, st.nn.Module]]:
        """Select layers based on parameter count"""
        layers = []

        # Classifier
        layers.append(("classifier_conv", model[1][1]))

        # Collect all conv layers with size estimates
        candidates = []
        feature_block = model[0]

        for i, layer in enumerate(feature_block):
            if isinstance(layer, st.nn.Conv2d):
                # Rough size estimate
                if hasattr(layer, "weight") and layer.weight is not None:
                    size = layer.weight.numel()
                    candidates.append((size, f"feature_{i}", layer))
            elif hasattr(layer, "__len__"):
                for j, sublayer in enumerate(layer):
                    if isinstance(sublayer, st.nn.Conv2d):
                        if hasattr(sublayer, "weight") and sublayer.weight is not None:
                            size = sublayer.weight.numel()
                            candidates.append((size, f"fire_{i}_{j}", sublayer))

        # Sort by size and take largest (or smallest)
        candidates.sort(reverse=True)  # Largest first

        for _, name, layer in candidates[: self.max_layers - 1]:
            layers.append((name, layer))

        return layers

    def get_description(self) -> str:
        priority = "recent-first" if self.prioritize_recent else "early-first"
        return f"Complex {self.strategy} selection (max {self.max_layers} layers, {priority})"


# How to use your heuristic:
if __name__ == "__main__":
    # Test your heuristic
    from helpers.models import squeezenet

    # Load a model
    model = squeezenet(pretrained=True, eval=True)

    # Create your heuristic
    heuristic = TemplateHeuristic(your_parameter=2)

    # Test layer selection
    selected_layers = heuristic.select_layers(model)

    print(f"Heuristic: {heuristic.get_description()}")
    print(f"Selected {len(selected_layers)} layers:")
    for name, layer in selected_layers:
        print(f"  {name}: {type(layer).__name__}")
    print(f"Complexity score: {heuristic.get_complexity_score()}")
    print(f"Valid selection: {heuristic.validate_selection(selected_layers)}")
