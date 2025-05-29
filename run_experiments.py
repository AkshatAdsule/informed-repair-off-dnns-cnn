#!/usr/bin/env python3
"""
Example script showing how to use the repair experiment framework
"""

from experiment_framework import RepairExperiment
from heuristics import (
    ClassifierOnlyHeuristic,
    LastNLayersHeuristic,
    FireModuleHeuristic,
    GradientBasedHeuristic,
    AdaptiveHeuristic,
    create_heuristic,
    list_all_heuristics,
)


def quick_test():
    """Quick test with just a few images and heuristics"""

    heuristics = [
        ClassifierOnlyHeuristic(),
        LastNLayersHeuristic(2),
    ]

    experiment = RepairExperiment(param_change_bound=15.0, margin=15.0)
    results = experiment.run_experiment(heuristics, num_images=2)

    experiment.print_summary(results)
    experiment.save_results(results, "quick_test_results.json")


def comprehensive_test():
    """More comprehensive test with different parameter settings"""

    # Test different layer selection strategies including new complex ones
    heuristics = [
        ClassifierOnlyHeuristic(),
        LastNLayersHeuristic(2),
        LastNLayersHeuristic(3),
        LastNLayersHeuristic(4),
        FireModuleHeuristic(1),
        FireModuleHeuristic(2),
        GradientBasedHeuristic(3),
        AdaptiveHeuristic("confidence_based", max_layers=4),
        AdaptiveHeuristic("architecture_aware", max_layers=5),
    ]

    # Test with different parameter bounds
    for bound in [5.0, 10.0, 15.0, 20.0]:
        print(f"\n{'=' * 60}")
        print(f"Testing with parameter bound: {bound}")
        print(f"{'=' * 60}")

        experiment = RepairExperiment(param_change_bound=bound, margin=15.0)
        results = experiment.run_experiment(heuristics, num_images=5)

        experiment.print_summary(results)
        experiment.save_results(results, f"comprehensive_bound_{bound}_results.json")


def margin_sensitivity_test():
    """Test how different margin values affect repair success"""

    heuristics = [
        ClassifierOnlyHeuristic(),
        AdaptiveHeuristic("layer_size_balanced", max_layers=3),
    ]

    for margin in [1.0, 5.0, 10.0, 15.0, 20.0]:
        print(f"\n{'=' * 60}")
        print(f"Testing with margin: {margin}")
        print(f"{'=' * 60}")

        experiment = RepairExperiment(param_change_bound=15.0, margin=margin)
        results = experiment.run_experiment(heuristics, num_images=5)

        experiment.print_summary(results)
        experiment.save_results(results, f"margin_{margin}_results.json")


def custom_heuristic_example():
    """Example of creating a custom heuristic using the base class"""

    from heuristics.base import LayerSelectionHeuristic
    import sytorch as st
    import random

    class RandomLayersHeuristic(LayerSelectionHeuristic):
        """Select random layers (example custom heuristic)"""

        def __init__(self, num_layers: int = 3, seed: int = 42):
            super().__init__(f"random_{num_layers}_layers")
            self.num_layers = num_layers
            self.seed = seed

        def select_layers(self, model):
            random.seed(self.seed)  # For reproducible results

            # Always include classifier
            layers = [("classifier_conv", model[1][1])]

            # Add random conv layers from features
            all_conv_layers = []
            feature_block = model[0]

            for i, layer in enumerate(feature_block):
                if isinstance(layer, st.nn.Conv2d):
                    all_conv_layers.append((f"features_{i}", layer))
                elif hasattr(layer, "__len__"):
                    for j, sublayer in enumerate(layer):
                        if isinstance(sublayer, st.nn.Conv2d):
                            all_conv_layers.append((f"features_{i}_{j}", sublayer))

            # Randomly sample additional layers
            if all_conv_layers and self.num_layers > 1:
                additional = min(self.num_layers - 1, len(all_conv_layers))
                random_layers = random.sample(all_conv_layers, additional)
                layers.extend(random_layers)

            return layers

        def get_description(self) -> str:
            return f"Randomly selects {self.num_layers} layers (seed={self.seed})"

        def get_complexity_score(self) -> int:
            return self.num_layers * 180000  # Random estimate

    # Test custom heuristic alongside existing ones
    heuristics = [
        ClassifierOnlyHeuristic(),
        RandomLayersHeuristic(2, seed=42),
        RandomLayersHeuristic(3, seed=123),
        GradientBasedHeuristic(2, include_classifier=True),
    ]

    experiment = RepairExperiment()
    results = experiment.run_experiment(heuristics, num_images=3)

    experiment.print_summary(results)
    experiment.save_results(results, "custom_heuristic_results.json")


def explore_heuristics():
    """Show all available heuristics and their descriptions"""
    print("Exploring available heuristics...")
    list_all_heuristics()

    print("\n" + "=" * 60)
    print("Testing heuristic creation with parameters:")
    print("=" * 60)

    # Show examples of creating heuristics with different parameters
    examples = [
        ("classifier_only", {}),
        ("last_n_layers", {"n": 4}),
        ("fire_module", {"num_modules": 2}),
        ("gradient_based", {"num_layers": 3, "include_classifier": False}),
        ("adaptive", {"strategy": "layer_size_balanced", "max_layers": 5}),
    ]

    for name, kwargs in examples:
        try:
            heuristic = create_heuristic(name, **kwargs)
            print(f"\n{name} with {kwargs}:")
            print(f"  -> {heuristic.get_description()}")
            print(f"  -> Complexity score: {heuristic.get_complexity_score()}")
        except Exception as e:
            print(f"\n{name}: Error - {e}")


def adaptive_strategies_test():
    """Test different adaptive strategies"""

    strategies = ["confidence_based", "architecture_aware", "layer_size_balanced"]
    heuristics = []

    for strategy in strategies:
        for max_layers in [2, 3, 4]:
            heuristics.append(AdaptiveHeuristic(strategy, max_layers))

    # Add baseline for comparison
    heuristics.insert(0, ClassifierOnlyHeuristic())

    experiment = RepairExperiment(param_change_bound=15.0, margin=15.0)
    results = experiment.run_experiment(heuristics, num_images=4)

    experiment.print_summary(results)
    experiment.save_results(results, "adaptive_strategies_results.json")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_type = sys.argv[1]

        if test_type == "quick":
            quick_test()
        elif test_type == "comprehensive":
            comprehensive_test()
        elif test_type == "margin":
            margin_sensitivity_test()
        elif test_type == "custom":
            custom_heuristic_example()
        elif test_type == "explore":
            explore_heuristics()
        elif test_type == "adaptive":
            adaptive_strategies_test()
        else:
            print(
                "Unknown test type. Options: quick, comprehensive, margin, custom, explore, adaptive"
            )
    else:
        print("Running quick test by default...")
        quick_test()
