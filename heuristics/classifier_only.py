"""
Classifier-only heuristic: Repair only the final classification layer.
"""

from typing import List, Tuple
import sytorch as st
from .base import LayerSelectionHeuristic


class ClassifierOnlyHeuristic(LayerSelectionHeuristic):
    """Select only the final classifier layer for repair"""

    def __init__(self):
        super().__init__("classifier_only")

    def select_layers(self, model: st.nn.Module) -> List[Tuple[str, st.nn.Module]]:
        """Select only the classifier convolutional layer"""
        return [("classifier_conv", model[1][1])]

    def get_description(self) -> str:
        return "Repairs only the final classifier convolutional layer"

    def get_complexity_score(self) -> int:
        # Typically the classifier layer has num_features * num_classes parameters
        # For SqueezeNet: 512 * 1000 = 512,000 parameters (roughly)
        return 512000
