"""
Layer selection heuristics for neural network repair.

This package contains various heuristics for selecting which layers
to make symbolic during the repair process.
"""

from .base import LayerSelectionHeuristic
from .classifier_only import ClassifierOnlyHeuristic
from .last_n_layers import LastNLayersHeuristic
from .fire_module import FireModuleHeuristic
from .gradient_based import GradientBasedHeuristic
from .adaptive import AdaptiveHeuristic

# List of all available heuristics for easy discovery
__all__ = [
    "LayerSelectionHeuristic",
    "ClassifierOnlyHeuristic",
    "LastNLayersHeuristic",
    "FireModuleHeuristic",
    "GradientBasedHeuristic",
    "AdaptiveHeuristic",
]

# Registry for automatic discovery
HEURISTIC_REGISTRY = {
    "classifier_only": ClassifierOnlyHeuristic,
    "last_n_layers": LastNLayersHeuristic,
    "fire_module": FireModuleHeuristic,
    "gradient_based": GradientBasedHeuristic,
    "adaptive": AdaptiveHeuristic,
}


def get_available_heuristics():
    """Get a list of all available heuristic names"""
    return list(HEURISTIC_REGISTRY.keys())


def create_heuristic(name: str, **kwargs):
    """
    Create a heuristic by name with parameters.

    Args:
        name: Name of the heuristic class
        **kwargs: Parameters to pass to the heuristic constructor

    Returns:
        Instantiated heuristic object

    Examples:
        heuristic = create_heuristic('last_n_layers', n=3)
        heuristic = create_heuristic('adaptive', strategy='confidence_based', max_layers=5)
        heuristic = create_heuristic('gradient_based', num_layers=4, include_classifier=True)
    """
    if name not in HEURISTIC_REGISTRY:
        available = ", ".join(get_available_heuristics())
        raise ValueError(f"Unknown heuristic '{name}'. Available: {available}")

    return HEURISTIC_REGISTRY[name](**kwargs)


def get_heuristic_info(name: str) -> str:
    """Get information about a heuristic"""
    if name not in HEURISTIC_REGISTRY:
        return f"Unknown heuristic: {name}"

    heuristic_class = HEURISTIC_REGISTRY[name]
    return heuristic_class.__doc__ or f"Heuristic: {name}"


def list_all_heuristics():
    """Print information about all available heuristics"""
    print("Available Heuristics:")
    print("=" * 50)

    for name in get_available_heuristics():
        info = get_heuristic_info(name)
        print(f"\n{name}:")
        print(f"  {info}")

        # Show example instantiation
        try:
            example = create_heuristic(name)
            print(f"  Example: {example.get_description()}")
        except Exception:
            print(f"  Example: create_heuristic('{name}')")
