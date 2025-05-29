"""
Fire module heuristic: Repair entire Fire modules in SqueezeNet.
"""

from typing import List, Tuple
import sytorch as st
from .base import LayerSelectionHeuristic


class FireModuleHeuristic(LayerSelectionHeuristic):
    """Select entire Fire modules for repair"""

    def __init__(self, num_modules: int = 1):
        super().__init__(f"fire_modules_{num_modules}")
        self.num_modules = num_modules

    def select_layers(self, model: st.nn.Module) -> List[Tuple[str, st.nn.Module]]:
        """Select all conv layers from the last N Fire modules plus classifier"""
        layers = []

        # Always include classifier
        layers.append(("classifier_conv", model[1][1]))

        # Add fire modules (working backwards)
        feature_block = model[0]
        fire_count = 0

        for i in range(len(feature_block) - 1, -1, -1):
            layer = feature_block[i]
            if hasattr(layer, "__len__") and len(layer) > 2:  # Likely a fire module
                # Add all conv layers in this fire module
                for j, sublayer in enumerate(layer):
                    if isinstance(sublayer, st.nn.Conv2d):
                        layers.append((f"fire_{i}_{j}", sublayer))

                fire_count += 1
                if fire_count >= self.num_modules:
                    break

        return layers

    def get_description(self) -> str:
        return f"Repairs {self.num_modules} Fire module(s) plus classifier (architecture-aware)"

    def get_complexity_score(self) -> int:
        # Each Fire module typically has 3 conv layers:
        # - squeeze layer (small)
        # - expand 1x1 (medium)
        # - expand 3x3 (medium)
        # Rough estimate: ~150k parameters per Fire module
        return (self.num_modules * 150000) + 512000  # Fire modules + classifier
