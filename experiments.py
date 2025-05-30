#!/usr/bin/env python3
"""
Experimental Setup - Heuristic Evaluation Script

This script implements the experimental framework described in the Experimental Setup PDF.
It evaluates different layer selection heuristics for neural network repair on SqueezeNet.

Heuristics tested (from the experimental setup):
1. Activation-Based: Selecting layers based on activation magnitude/variance
2. Gradient/Sensitivity-Based: Layers with highest parameter sensitivity
3. Feature-Similarity Based: Layers with similar internal representations
4. Layer Type/Position: Structural heuristics (penultimate, first FC, etc.)
5. Neuron/Path Contribution: Layers with highest contribution density

Skipping adversarial heuristics as requested.
"""

import sys
import os
import json
import time
from typing import List, Dict, Any
from experiment_framework import RepairExperiment
from heuristics import (
    # Activation-based heuristics
    ActivationMagnitudeHeuristic,
    ActivationVarianceHeuristic,
    # Sensitivity-based heuristics
    GradientNormHeuristic,
    ParameterSensitivityHeuristic,
    OutputSensitivityHeuristic,
    # Feature similarity heuristics
    FeatureConvergenceHeuristic,
    FeatureDivergenceHeuristic,
    BottleneckHeuristic,
    # Layer position heuristics
    PenultimateLayerHeuristic,
    FirstFCHeuristic,
    TransitionLayerHeuristic,
    # Neuron contribution heuristics
    MagnitudeContributionHeuristic,
    VarianceContributionHeuristic,
    ConsistencyContributionHeuristic,
    PathStrengthHeuristic,
    # For comparison
    ClassifierOnlyHeuristic,
    LastNLayersHeuristic,
)


def create_heuristics() -> List:
    """
    Create all heuristics defined in the Experimental Setup.

    Returns:
        List of heuristic instances organized by category
    """

    heuristics = []

    # === Activation-Based Heuristics ===
    print("Creating Activation-Based heuristics...")
    heuristics.extend(
        [
            ActivationMagnitudeHeuristic(num_layers=1, include_classifier=True),
            ActivationMagnitudeHeuristic(num_layers=2, include_classifier=True),
            ActivationMagnitudeHeuristic(num_layers=3, include_classifier=True),
            ActivationVarianceHeuristic(num_layers=1, include_classifier=True),
            ActivationVarianceHeuristic(num_layers=2, include_classifier=True),
            ActivationVarianceHeuristic(num_layers=3, include_classifier=True),
        ]
    )

    # === Gradient/Sensitivity-Based Heuristics ===
    print("Creating Sensitivity-Based heuristics...")
    heuristics.extend(
        [
            GradientNormHeuristic(num_layers=1, include_classifier=True),
            GradientNormHeuristic(num_layers=2, include_classifier=True),
            GradientNormHeuristic(num_layers=3, include_classifier=True),
            ParameterSensitivityHeuristic(num_layers=1, include_classifier=True),
            ParameterSensitivityHeuristic(num_layers=2, include_classifier=True),
            ParameterSensitivityHeuristic(num_layers=3, include_classifier=True),
            OutputSensitivityHeuristic(num_layers=1, include_classifier=True),
            OutputSensitivityHeuristic(num_layers=2, include_classifier=True),
            OutputSensitivityHeuristic(num_layers=3, include_classifier=True),
        ]
    )

    # === Feature-Similarity Based Heuristics ===
    print("Creating Feature-Similarity heuristics...")
    heuristics.extend(
        [
            FeatureConvergenceHeuristic(num_layers=1, include_classifier=True),
            FeatureConvergenceHeuristic(num_layers=2, include_classifier=True),
            FeatureConvergenceHeuristic(num_layers=3, include_classifier=True),
            FeatureDivergenceHeuristic(num_layers=1, include_classifier=True),
            FeatureDivergenceHeuristic(num_layers=2, include_classifier=True),
            FeatureDivergenceHeuristic(num_layers=3, include_classifier=True),
            BottleneckHeuristic(num_layers=1, include_classifier=True),
            BottleneckHeuristic(num_layers=2, include_classifier=True),
            BottleneckHeuristic(num_layers=3, include_classifier=True),
        ]
    )

    # === Layer Type/Position Heuristics ===
    print("Creating Layer Position heuristics...")
    heuristics.extend(
        [
            PenultimateLayerHeuristic(num_layers=1),
            PenultimateLayerHeuristic(num_layers=2),
            PenultimateLayerHeuristic(num_layers=3),
            FirstFCHeuristic(num_layers=1),
            FirstFCHeuristic(num_layers=2),
            FirstFCHeuristic(num_layers=3),
            TransitionLayerHeuristic(num_layers=1),
            TransitionLayerHeuristic(num_layers=2),
            TransitionLayerHeuristic(num_layers=3),
        ]
    )

    # === Neuron/Path Contribution Profiling ===
    print("Creating Neuron Contribution heuristics...")
    heuristics.extend(
        [
            MagnitudeContributionHeuristic(num_layers=1, include_classifier=True),
            MagnitudeContributionHeuristic(num_layers=2, include_classifier=True),
            MagnitudeContributionHeuristic(num_layers=3, include_classifier=True),
            VarianceContributionHeuristic(num_layers=1, include_classifier=True),
            VarianceContributionHeuristic(num_layers=2, include_classifier=True),
            VarianceContributionHeuristic(num_layers=3, include_classifier=True),
            ConsistencyContributionHeuristic(num_layers=1, include_classifier=True),
            ConsistencyContributionHeuristic(num_layers=2, include_classifier=True),
            ConsistencyContributionHeuristic(num_layers=3, include_classifier=True),
            PathStrengthHeuristic(num_layers=1, include_classifier=True),
            PathStrengthHeuristic(num_layers=2, include_classifier=True),
            PathStrengthHeuristic(num_layers=3, include_classifier=True),
        ]
    )

    # === Baseline Comparisons ===
    print("Creating baseline heuristics...")
    heuristics.extend(
        [
            ClassifierOnlyHeuristic(),
            LastNLayersHeuristic(n=1),
            LastNLayersHeuristic(n=2),
            LastNLayersHeuristic(n=3),
        ]
    )

    print(f"Created {len(heuristics)} heuristics total")
    return heuristics


def run_comprehensive_evaluation():
    """
    Run comprehensive evaluation following experimental setup.

    This evaluates all heuristics across multiple edit set sizes and parameters.
    """
    print("=" * 80)
    print("Comprehensive Heuristic Evaluation")
    print("=" * 80)

    # Create experiment framework
    experiment = RepairExperiment(param_change_bound=15.0, margin=15.0)

    # Create all heuristics
    heuristics = create_heuristics()

    # Test different edit set sizes (as mentioned in experimental setup)
    edit_set_sizes = [1, 3, 5, 10]

    all_results = []

    for edit_size in edit_set_sizes:
        print(f"\n" + "=" * 50)
        print(f"Testing with {edit_size} images from edit set")
        print("=" * 50)

        start_time = time.time()
        results = experiment.run_experiment(
            heuristics=heuristics,
            num_images=edit_size,
            edit_set_path="data/edit_sets/squeezenet_edit_dataset.pt",
        )

        # Add edit set size to results
        for result in results:
            result["edit_set_size"] = edit_size

        all_results.extend(results)

        elapsed = time.time() - start_time
        print(f"Completed {edit_size}-image evaluation in {elapsed:.1f}s")

        # Save intermediate results
        filename = f"results_size_{edit_size}.json"
        experiment.save_results(results, filename)
        print(f"Saved results to {filename}")

    # Save comprehensive results
    experiment.save_results(all_results, "comprehensive_results.json")

    # Generate summary report
    generate_report(all_results)

    return all_results


def run_quick_evaluation():
    """
    Run a quick evaluation for testing the framework.
    """
    print("=" * 60)
    print("Quick Heuristic Evaluation")
    print("=" * 60)

    experiment = RepairExperiment(param_change_bound=15.0, margin=15.0)

    # Test a subset of representative heuristics
    heuristics = [
        # One from each category
        ActivationMagnitudeHeuristic(num_layers=2),
        GradientNormHeuristic(num_layers=2),
        FeatureConvergenceHeuristic(num_layers=2),
        PenultimateLayerHeuristic(num_layers=2),
        MagnitudeContributionHeuristic(num_layers=2),
        # Baselines
        ClassifierOnlyHeuristic(),
        LastNLayersHeuristic(n=2),
    ]

    print(f"Testing {len(heuristics)} representative heuristics...")

    results = experiment.run_experiment(
        heuristics=heuristics,
        num_images=3,  # Small set for quick testing
        edit_set_path="data/edit_sets/squeezenet_edit_dataset.pt",
    )

    experiment.save_results(results, "quick_results.json")
    experiment.print_summary(results)

    return results


def run_scalability_analysis():
    """
    Run scalability analysis as mentioned in the experimental setup metrics.
    """
    print("=" * 60)
    print("Scalability Analysis")
    print("=" * 60)

    experiment = RepairExperiment(param_change_bound=15.0, margin=15.0)

    # Test scalability with different layer counts
    layer_counts = [1, 2, 3, 4]

    scalability_results = []

    for num_layers in layer_counts:
        print(f"\nTesting scalability with {num_layers} layers...")

        # Test representative heuristics with different layer counts
        test_heuristics = [
            ActivationMagnitudeHeuristic(num_layers=num_layers),
            GradientNormHeuristic(num_layers=num_layers),
            MagnitudeContributionHeuristic(num_layers=num_layers),
        ]

        start_time = time.time()
        results = experiment.run_experiment(
            heuristics=test_heuristics,
            num_images=5,
            edit_set_path="data/edit_sets/squeezenet_edit_dataset.pt",
        )
        elapsed = time.time() - start_time

        # Add scalability metrics
        for result in results:
            result["num_layers_tested"] = num_layers
            result["total_time_per_layer"] = elapsed / num_layers

        scalability_results.extend(results)
        print(f"  Completed in {elapsed:.1f}s ({elapsed / num_layers:.1f}s per layer)")

    # Save scalability results
    experiment.save_results(scalability_results, "scalability_results.json")

    return scalability_results


def generate_report(results: List[Dict[str, Any]]):
    """
    Generate a comprehensive report of the experimental results.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENTAL RESULTS REPORT")
    print("=" * 80)

    # Group results by heuristic category
    categories = {
        "Activation-Based": ["activation_magnitude", "activation_variance"],
        "Sensitivity-Based": [
            "gradient_norm",
            "parameter_sensitivity",
            "output_sensitivity",
        ],
        "Feature-Similarity": [
            "feature_convergence",
            "feature_divergence",
            "bottleneck",
        ],
        "Position-Based": ["penultimate", "first_fc", "transition"],
        "Contribution-Based": [
            "magnitude_contribution",
            "variance_contribution",
            "consistency_contribution",
            "path_strength",
        ],
        "Baselines": ["classifier_only", "last_n_layers"],
    }

    for category, heuristic_patterns in categories.items():
        print(f"\n{category} Heuristics:")
        print("-" * 40)

        category_results = []
        for result in results:
            heuristic_name = result["heuristic_name"].lower()
            if any(pattern in heuristic_name for pattern in heuristic_patterns):
                category_results.append(result)

        if category_results:
            # Calculate success rate
            successful = sum(1 for r in category_results if r["repair_successful"])
            total = len(category_results)
            success_rate = (successful / total) * 100 if total > 0 else 0

            print(f"  Success Rate: {successful}/{total} ({success_rate:.1f}%)")

            # Average solve time for successful repairs
            successful_results = [r for r in category_results if r["repair_successful"]]
            if successful_results:
                avg_time = sum(r["solve_time"] for r in successful_results) / len(
                    successful_results
                )
                print(f"  Avg Solve Time: {avg_time:.2f}s")

            # Complexity analysis
            complexities = [r.get("complexity_score", 0) for r in category_results]
            if complexities:
                avg_complexity = sum(complexities) / len(complexities)
                print(f"  Avg Complexity: {avg_complexity:,.0f} parameters")
        else:
            print(f"  No results found")

    # Overall statistics
    print(f"\n{'Overall Statistics':=^60}")
    total_experiments = len(results)
    successful_repairs = sum(1 for r in results if r["repair_successful"])
    overall_success_rate = (
        (successful_repairs / total_experiments) * 100 if total_experiments > 0 else 0
    )

    print(f"Total experiments: {total_experiments}")
    print(f"Successful repairs: {successful_repairs}")
    print(f"Overall success rate: {overall_success_rate:.1f}%")

    if successful_repairs > 0:
        successful_results = [r for r in results if r["repair_successful"]]
        avg_solve_time = sum(r["solve_time"] for r in successful_results) / len(
            successful_results
        )
        print(f"Average solve time: {avg_solve_time:.2f}s")

    # Save detailed report
    report_data = {
        "experiment_name": "Heuristic Evaluation",
        "total_experiments": total_experiments,
        "successful_repairs": successful_repairs,
        "overall_success_rate": overall_success_rate,
        "category_breakdown": {},
    }

    for category, heuristic_patterns in categories.items():
        category_results = [
            r
            for r in results
            if any(
                pattern in r["heuristic_name"].lower() for pattern in heuristic_patterns
            )
        ]
        if category_results:
            successful = sum(1 for r in category_results if r["repair_successful"])
            total = len(category_results)
            success_rate = (successful / total) * 100 if total > 0 else 0

            report_data["category_breakdown"][category] = {
                "total_experiments": total,
                "successful_repairs": successful,
                "success_rate": success_rate,
            }

    with open("experiment_report.json", "w") as f:
        json.dump(report_data, f, indent=2)

    print(f"\nDetailed report saved to: experiment_report.json")


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description="Heuristic Evaluation")
    parser.add_argument(
        "--mode",
        choices=["quick", "comprehensive", "scalability"],
        default="quick",
        help="Evaluation mode",
    )
    parser.add_argument(
        "--list-heuristics", action="store_true", help="List all available heuristics"
    )

    args = parser.parse_args()

    if args.list_heuristics:
        print("Experimental Setup Heuristics:")
        print("=" * 50)
        heuristics = create_heuristics()
        for i, h in enumerate(heuristics, 1):
            print(f"{i:2d}. {h.name}: {h.get_description()}")
        return

    if args.mode == "quick":
        results = run_quick_evaluation()
    elif args.mode == "comprehensive":
        results = run_comprehensive_evaluation()
    elif args.mode == "scalability":
        results = run_scalability_analysis()
    else:
        print(f"Unknown mode: {args.mode}")
        return

    print(f"\nExperiment completed. Processed {len(results)} repair attempts.")


if __name__ == "__main__":
    main()
