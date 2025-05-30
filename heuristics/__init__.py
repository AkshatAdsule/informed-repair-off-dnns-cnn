"""
Heuristics package for neural network repair.
Contains various layer selection heuristics for repair experiments.
"""

# Base class
from .base import LayerSelectionHeuristic

# Original heuristics
from .classifier_only import ClassifierOnlyHeuristic
from .last_n_layers import LastNLayersHeuristic
from .fire_module import FireModuleHeuristic
from .gradient_based import GradientBasedHeuristic
from .adaptive import AdaptiveHeuristic

# Experimental Setup heuristics
from .activation_based import (
    ActivationBasedHeuristic,
    ActivationMagnitudeHeuristic,
    ActivationVarianceHeuristic,
)
from .sensitivity_based import (
    SensitivityBasedHeuristic,
    GradientNormHeuristic,
    ParameterSensitivityHeuristic,
    OutputSensitivityHeuristic,
)
from .feature_similarity import (
    FeatureSimilarityHeuristic,
    FeatureConvergenceHeuristic,
    FeatureDivergenceHeuristic,
    BottleneckHeuristic,
)
from .layer_position import (
    LayerPositionHeuristic,
    PenultimateLayerHeuristic,
    FirstFCHeuristic,
    TransitionLayerHeuristic,
)
from .neuron_contribution import (
    NeuronContributionHeuristic,
    MagnitudeContributionHeuristic,
    VarianceContributionHeuristic,
    ConsistencyContributionHeuristic,
    PathStrengthHeuristic,
)

# Registry of all available heuristics
HEURISTIC_REGISTRY = {
    # Original heuristics
    "classifier_only": ClassifierOnlyHeuristic,
    "last_n_layers": LastNLayersHeuristic,
    "fire_module": FireModuleHeuristic,
    "gradient_based": GradientBasedHeuristic,
    "adaptive": AdaptiveHeuristic,
    # Activation-based heuristics
    "activation_based": ActivationBasedHeuristic,
    "activation_magnitude": ActivationMagnitudeHeuristic,
    "activation_variance": ActivationVarianceHeuristic,
    # Sensitivity-based heuristics
    "sensitivity_based": SensitivityBasedHeuristic,
    "gradient_norm": GradientNormHeuristic,
    "parameter_sensitivity": ParameterSensitivityHeuristic,
    "output_sensitivity": OutputSensitivityHeuristic,
    # Feature similarity heuristics
    "feature_similarity": FeatureSimilarityHeuristic,
    "feature_convergence": FeatureConvergenceHeuristic,
    "feature_divergence": FeatureDivergenceHeuristic,
    "bottleneck": BottleneckHeuristic,
    # Layer position heuristics
    "layer_position": LayerPositionHeuristic,
    "penultimate": PenultimateLayerHeuristic,
    "first_fc": FirstFCHeuristic,
    "transition": TransitionLayerHeuristic,
    # Neuron contribution heuristics
    "neuron_contribution": NeuronContributionHeuristic,
    "magnitude_contribution": MagnitudeContributionHeuristic,
    "variance_contribution": VarianceContributionHeuristic,
    "consistency_contribution": ConsistencyContributionHeuristic,
    "path_strength": PathStrengthHeuristic,
}


def get_available_heuristics():
    """Get list of all available heuristic names"""
    return list(HEURISTIC_REGISTRY.keys())


def create_heuristic(name: str, **kwargs):
    """
    Create a heuristic by name with optional parameters.

    Args:
        name: Name of the heuristic (from HEURISTIC_REGISTRY)
        **kwargs: Additional parameters to pass to the heuristic constructor

    Returns:
        LayerSelectionHeuristic: Instance of the requested heuristic

    Raises:
        ValueError: If heuristic name is not recognized

    Examples:
        >>> h1 = create_heuristic("classifier_only")
        >>> h2 = create_heuristic("last_n_layers", n=5)
        >>> h3 = create_heuristic("activation_magnitude", num_layers=2)
    """
    if name not in HEURISTIC_REGISTRY:
        available = ", ".join(get_available_heuristics())
        raise ValueError(f"Unknown heuristic '{name}'. Available: {available}")

    heuristic_class = HEURISTIC_REGISTRY[name]
    return heuristic_class(**kwargs)


def get_heuristic_info(name: str) -> str:
    """Get description of a specific heuristic"""
    if name not in HEURISTIC_REGISTRY:
        return f"Unknown heuristic: {name}"

    heuristic = create_heuristic(name)
    return f"{name}: {heuristic.get_description()}"


def list_all_heuristics():
    """Print information about all available heuristics"""
    print("Available Heuristics:")
    print("=" * 50)

    categories = {
        "Basic": ["classifier_only", "last_n_layers", "fire_module"],
        "Adaptive": ["gradient_based", "adaptive"],
        "Activation-Based": [
            "activation_based",
            "activation_magnitude",
            "activation_variance",
        ],
        "Sensitivity-Based": [
            "sensitivity_based",
            "gradient_norm",
            "parameter_sensitivity",
            "output_sensitivity",
        ],
        "Feature Similarity": [
            "feature_similarity",
            "feature_convergence",
            "feature_divergence",
            "bottleneck",
        ],
        "Position-Based": ["layer_position", "penultimate", "first_fc", "transition"],
        "Contribution-Based": [
            "neuron_contribution",
            "magnitude_contribution",
            "variance_contribution",
            "consistency_contribution",
            "path_strength",
        ],
    }

    for category, heuristics in categories.items():
        print(f"\n{category}:")
        for h_name in heuristics:
            if h_name in HEURISTIC_REGISTRY:
                heuristic = create_heuristic(h_name)
                print(f"  • {h_name}: {heuristic.get_description()}")


# Make key classes available at package level
__all__ = [
    # Base
    "LayerSelectionHeuristic",
    # Original
    "ClassifierOnlyHeuristic",
    "LastNLayersHeuristic",
    "FireModuleHeuristic",
    "GradientBasedHeuristic",
    "AdaptiveHeuristic",
    # Experimental Setup heuristics
    "ActivationBasedHeuristic",
    "ActivationMagnitudeHeuristic",
    "ActivationVarianceHeuristic",
    "SensitivityBasedHeuristic",
    "GradientNormHeuristic",
    "ParameterSensitivityHeuristic",
    "OutputSensitivityHeuristic",
    "FeatureSimilarityHeuristic",
    "FeatureConvergenceHeuristic",
    "FeatureDivergenceHeuristic",
    "BottleneckHeuristic",
    "LayerPositionHeuristic",
    "PenultimateLayerHeuristic",
    "FirstFCHeuristic",
    "TransitionLayerHeuristic",
    "NeuronContributionHeuristic",
    "MagnitudeContributionHeuristic",
    "VarianceContributionHeuristic",
    "ConsistencyContributionHeuristic",
    "PathStrengthHeuristic",
    # Utility functions
    "get_available_heuristics",
    "create_heuristic",
    "get_heuristic_info",
    "list_all_heuristics",
    "HEURISTIC_REGISTRY",
]
