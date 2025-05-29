"""
Last N layers heuristic: Repair the last N convolutional layers.
"""

from typing import List, Tuple
import sytorch as st
from .base import LayerSelectionHeuristic


class LastNLayersHeuristic(LayerSelectionHeuristic):
    """Select the last N convolutional layers for repair"""

    def __init__(self, n: int = 2):
        super().__init__(f"last_{n}_layers")
        self.n = n

    def select_layers(self, model: st.nn.Module) -> List[Tuple[str, st.nn.Module]]:
        """Select the last N convolutional layers, including classifier"""
        layers = []

        # Always include classifier
        layers.append(("classifier_conv", model[1][1]))

        # Add last few feature layers (working backwards)
        if self.n > 1:
            feature_block = model[0]
            conv_layers = []

            # Find conv layers in reverse order
            for i in range(len(feature_block) - 1, -1, -1):
                layer = feature_block[i]
                if hasattr(layer, "__len__") and len(layer) > 0:
                    # Fire module - check for conv layers
                    for j, sublayer in enumerate(layer):
                        if isinstance(sublayer, st.nn.Conv2d):
                            conv_layers.append((f"features_{i}_{j}", sublayer))
                elif isinstance(layer, st.nn.Conv2d):
                    conv_layers.append((f"features_{i}", layer))

                if len(conv_layers) >= self.n - 1:
                    break

            layers.extend(conv_layers[: self.n - 1])

        return layers

    def get_description(self) -> str:
        return f"Repairs the last {self.n} convolutional layers (including classifier)"

    def get_complexity_score(self) -> int:
        # Rough estimate: each layer might have ~50k-500k parameters
        # More layers = higher complexity
        return self.n * 200000  # Conservative middle estimate
