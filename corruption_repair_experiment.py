#!/usr/bin/env python3
"""
Corruption Repair Experiment - Compare Activation-Based vs Raw Repair Heuristics

This experiment compares activation-based heuristics against "raw-dogging" repair approaches
on corrupted king penguin images.

Setup:
- Repair on: king_penguin_test_fixed_edit_dataset.pt (12 corrupted images)
- Evaluate on: king_penguin_corrupted_fixed_edit_dataset.pt (126 corrupted images)
"""

import sys
import os
import json
import time
import torch
from typing import List, Dict, Any
from experiment_framework import RepairExperiment
from heuristics import (
    # Activation-based heuristics
    ActivationMagnitudeHeuristic,
    ActivationVarianceHeuristic,
    # Raw repair heuristics
    RawRepairHeuristic,
    ConservativeRawRepairHeuristic,
    LayerSubsetRawRepairHeuristic,
    # For comparison
    ClassifierOnlyHeuristic,
    LastNLayersHeuristic,
)


class CorruptionRepairExperiment(RepairExperiment):
    """Extended experiment framework for corruption repair evaluation"""

    def __init__(self, param_change_bound: float = 15.0, margin: float = 15.0):
        super().__init__(param_change_bound, margin)

    def run_single_repair_with_custom_bounds(
        self, model, image, true_label, heuristic, custom_param_bound: float = None
    ) -> Dict[str, Any]:
        """Run repair with custom parameter bounds for raw repair heuristics"""

        import sytorch as st

        # Setup solver and model
        solver = st.GurobiSolver()
        N = model.deepcopy().to(solver).repair()

        # Get layers to make symbolic
        layers_to_repair = heuristic.select_layers(N)

        # Validate the selection
        if not heuristic.validate_selection(layers_to_repair):
            raise ValueError(f"Invalid layer selection from heuristic {heuristic.name}")

        # Use custom bounds for raw repair heuristics
        if hasattr(heuristic, "param_bound") and custom_param_bound is None:
            param_bound = heuristic.param_bound
        else:
            param_bound = custom_param_bound or self.param_change_bound

        # Make selected layers symbolic
        for layer_name, layer in layers_to_repair:
            if hasattr(layer, "weight") and layer.weight is not None:
                layer.weight.requires_symbolic_(lb=-param_bound, ub=param_bound)
            if hasattr(layer, "bias") and layer.bias is not None:
                layer.bias.requires_symbolic_(lb=-param_bound, ub=param_bound)

        # Get original prediction
        with st.no_grad():
            original_output = model(image)
            _, original_pred = torch.max(original_output, 1)

        # Define constraints
        with st.no_symbolic(), st.no_grad():
            reference_output = N(image)

        symbolic_output = N(image)
        true_class_logit = symbolic_output[0, true_label]

        constraints = []
        for i in range(1000):  # ImageNet classes
            if i != true_label:
                constraints.append(
                    true_class_logit >= symbolic_output[0, i] + self.margin
                )

        # Solve
        param_deltas = N.parameter_deltas(concat=True)
        output_deltas = (symbolic_output - reference_output).flatten().alias()
        objective = st.cat([output_deltas, param_deltas]).norm_ub("linf+l1_normalized")

        success = solver.solve(*constraints, minimize=objective)

        repaired_pred = None
        if success:
            N.update_()
            N.repair(False)
            N.eval()

            with st.no_grad():
                repaired_output = N(image)
                _, repaired_pred = torch.max(repaired_output, 1)

        return {
            "heuristic": heuristic.name,
            "heuristic_description": heuristic.get_description(),
            "complexity_score": heuristic.get_complexity_score(),
            "layers_repaired": [name for name, _ in layers_to_repair],
            "num_layers": len(layers_to_repair),
            "param_bound_used": param_bound,
            "original_pred": original_pred.item(),
            "true_label": true_label,
            "success": success,
            "repaired_pred": repaired_pred.item()
            if repaired_pred is not None
            else None,
            "repair_correct": repaired_pred.item() == true_label
            if repaired_pred is not None
            else False,
        }

    def evaluate_on_set(
        self, model, eval_set_path: str, heuristics: List
    ) -> List[Dict[str, Any]]:
        """Evaluate repaired model on a separate evaluation set"""
        eval_set = self.load_edit_set(eval_set_path)
        results = []

        print(f"Evaluating on {len(eval_set['images'])} images from {eval_set_path}")

        for img_idx in range(len(eval_set["images"])):
            image = eval_set["images"][img_idx].unsqueeze(0)
            true_label = eval_set["labels"][img_idx].item()

            # Get original (corrupted) prediction
            with torch.no_grad():
                original_output = model(image)
                _, original_pred = torch.max(original_output, 1)

            # Get metadata if available
            metadata = {}
            if eval_set["metadata"] and img_idx < len(eval_set["metadata"]):
                metadata = eval_set["metadata"][img_idx]

            results.append(
                {
                    "eval_image_idx": img_idx,
                    "true_label": true_label,
                    "original_pred": original_pred.item(),
                    "is_correct": original_pred.item() == true_label,
                    "corruption_type": metadata.get("corruption_type", "unknown"),
                    "corruption_severity": metadata.get("corruption_severity", 0),
                    "original_correct": metadata.get("original_is_correct", False),
                }
            )

        return results

    def run_batch_repair_with_custom_bounds(
        self, model, images, true_labels, heuristic, custom_param_bound: float = None
    ) -> Dict[str, Any]:
        """Run repair with custom parameter bounds across a batch of images"""

        import sytorch as st

        # Setup solver and model
        solver = st.GurobiSolver()
        N = model.deepcopy().to(solver).repair()

        # Get layers to make symbolic
        layers_to_repair = heuristic.select_layers(N)

        # Validate the selection
        if not heuristic.validate_selection(layers_to_repair):
            raise ValueError(f"Invalid layer selection from heuristic {heuristic.name}")

        # Use custom bounds for raw repair heuristics
        if hasattr(heuristic, "param_bound") and custom_param_bound is None:
            param_bound = heuristic.param_bound
        else:
            param_bound = custom_param_bound or self.param_change_bound

        # Make selected layers symbolic
        for layer_name, layer in layers_to_repair:
            if hasattr(layer, "weight") and layer.weight is not None:
                layer.weight.requires_symbolic_(lb=-param_bound, ub=param_bound)
            if hasattr(layer, "bias") and layer.bias is not None:
                layer.bias.requires_symbolic_(lb=-param_bound, ub=param_bound)

        # Get original predictions for the batch
        with st.no_grad():
            original_outputs = model(images)
            _, original_preds = torch.max(original_outputs, 1)

        # Define constraints for all images in the batch
        with st.no_symbolic(), st.no_grad():
            reference_outputs = N(images)

        symbolic_outputs = N(images)

        constraints = []

        # Create constraints for each image in the batch
        for i, true_label in enumerate(true_labels):
            true_class_logit = symbolic_outputs[i, true_label]

            # For each image, true class should beat all other classes
            for j in range(1000):  # ImageNet classes
                if j != true_label:
                    constraints.append(
                        true_class_logit >= symbolic_outputs[i, j] + self.margin
                    )

        # Solve with batch objective
        param_deltas = N.parameter_deltas(concat=True)
        output_deltas = (symbolic_outputs - reference_outputs).flatten().alias()
        objective = st.cat([output_deltas, param_deltas]).norm_ub("linf+l1_normalized")

        success = solver.solve(*constraints, minimize=objective)

        repaired_preds = None
        if success:
            N.update_()
            N.repair(False)
            N.eval()

            with st.no_grad():
                repaired_outputs = N(images)
                _, repaired_preds = torch.max(repaired_outputs, 1)

        return {
            "heuristic": heuristic.name,
            "heuristic_description": heuristic.get_description(),
            "complexity_score": heuristic.get_complexity_score(),
            "layers_repaired": [name for name, _ in layers_to_repair],
            "num_layers": len(layers_to_repair),
            "param_bound_used": param_bound,
            "batch_size": len(images),
            "original_preds": original_preds.tolist(),
            "true_labels": true_labels.tolist(),
            "success": success,
            "repaired_preds": repaired_preds.tolist()
            if repaired_preds is not None
            else None,
            "repair_correct": [
                pred == true
                for pred, true in zip(repaired_preds.tolist(), true_labels.tolist())
            ]
            if repaired_preds is not None
            else [False] * len(true_labels),
            "repaired_model": N
            if success
            else None,  # Return the repaired model for evaluation
        }

    def run_batch_repair_experiment(
        self,
        heuristics: List,
        edit_set_path: str = "data/edit_sets/squeezenet_edit_dataset.pt",
    ) -> List[Dict[str, Any]]:
        """Run repair experiments using batch repair across the whole dataset"""

        print("Loading model and edit set...")
        model = self.load_model()
        edit_set = self.load_edit_set(edit_set_path)

        # Prepare batch data
        images = edit_set["images"]
        true_labels = edit_set["labels"]

        print(f"Loaded {len(images)} images for batch repair")

        results = []

        # Test each heuristic on the entire batch
        for heuristic in heuristics:
            print(f"\nTesting {heuristic.name} on batch of {len(images)} images...")

            if edit_set["metadata"]:
                print(f"  Dataset contains corrupted images with metadata")

            try:
                # Run batch repair
                result = self.run_batch_repair_with_custom_bounds(
                    model, images, true_labels, heuristic
                )
                results.append(result)

                # Analyze batch results
                if result["success"]:
                    correct_repairs = sum(result["repair_correct"])
                    total_images = result["batch_size"]
                    success_rate = correct_repairs / total_images

                    print(
                        f"    SUCCESS: {correct_repairs}/{total_images} images repaired correctly ({success_rate:.1%})"
                    )
                    print(
                        f"    Param bound: ±{result['param_bound_used']}, Layers: {result['num_layers']}"
                    )

                    # Show per-image details
                    for i, (orig, repaired, correct) in enumerate(
                        zip(
                            result["original_preds"],
                            result["repaired_preds"],
                            result["repair_correct"],
                        )
                    ):
                        status = "✓" if correct else "✗"
                        print(
                            f"      Image {i}: {orig} -> {repaired} (target: {true_labels[i].item()}) {status}"
                        )

                else:
                    print(f"    FAILED: Could not find solution")
                    print(
                        f"    Param bound: ±{result['param_bound_used']}, Layers: {result['num_layers']}"
                    )

            except Exception as e:
                print(f"    ERROR: {e}")
                results.append(
                    {
                        "heuristic": heuristic.name,
                        "success": False,
                        "error": str(e),
                        "batch_size": len(images),
                    }
                )

        return results

    def evaluate_repaired_models(
        self, repair_results: List[Dict[str, Any]], eval_set_path: str
    ) -> List[Dict[str, Any]]:
        """Evaluate repaired models on the evaluation set"""

        eval_set = self.load_edit_set(eval_set_path)
        eval_images = eval_set["images"]
        eval_labels = eval_set["labels"]

        print(
            f"\nEvaluating repaired models on {len(eval_images)} evaluation images..."
        )

        evaluation_results = []

        for repair_result in repair_results:
            if not repair_result.get("success", False):
                continue

            repaired_model = repair_result.get("repaired_model")
            if repaired_model is None:
                continue

            heuristic_name = repair_result.get("heuristic", "unknown")
            print(f"  Evaluating {heuristic_name}...")

            # Evaluate on the full evaluation set
            correct_predictions = 0
            total_predictions = len(eval_images)

            with torch.no_grad():
                repaired_model.eval()
                eval_outputs = repaired_model(eval_images)
                _, eval_preds = torch.max(eval_outputs, 1)

                correct_predictions = (eval_preds == eval_labels).sum().item()

            eval_accuracy = correct_predictions / total_predictions

            # Compare with repair set performance
            repair_correct = sum(repair_result.get("repair_correct", []))
            repair_total = repair_result.get("batch_size", 0)
            repair_accuracy = repair_correct / repair_total if repair_total > 0 else 0

            eval_result = {
                "heuristic": heuristic_name,
                "repair_accuracy": repair_accuracy,
                "eval_accuracy": eval_accuracy,
                "eval_correct": correct_predictions,
                "eval_total": total_predictions,
                "generalization_score": eval_accuracy / repair_accuracy
                if repair_accuracy > 0
                else 0,
                "num_layers": repair_result.get("num_layers", 0),
                "param_bound": repair_result.get("param_bound_used", 0),
            }

            evaluation_results.append(eval_result)

            print(f"    Repair accuracy: {repair_accuracy:.1%}")
            print(f"    Eval accuracy: {eval_accuracy:.1%}")
            print(f"    Generalization: {eval_result['generalization_score']:.2f}x")

        return evaluation_results


def create_comparison_heuristics() -> List:
    """Create heuristics for activation-based vs raw repair comparison"""

    heuristics = []

    print("Creating Activation-Based heuristics...")
    # Activation-based heuristics (the "smart" approaches)
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

    print("Creating Raw Repair heuristics...")
    # Raw repair heuristics (the "raw-dogging" approaches)
    heuristics.extend(
        [
            RawRepairHeuristic(param_bound=1000.0, include_classifier=True),
            RawRepairHeuristic(param_bound=500.0, include_classifier=True),
            ConservativeRawRepairHeuristic(param_bound=100.0, include_classifier=True),
            ConservativeRawRepairHeuristic(param_bound=50.0, include_classifier=True),
            LayerSubsetRawRepairHeuristic(
                param_bound=200.0, exclude_types=["BatchNorm"]
            ),
        ]
    )

    print("Creating baseline heuristics...")
    # Baseline comparisons
    heuristics.extend(
        [
            ClassifierOnlyHeuristic(),
            LastNLayersHeuristic(n=1),
            LastNLayersHeuristic(n=3),
            LastNLayersHeuristic(n=5),
        ]
    )

    print(f"Created {len(heuristics)} heuristics total")
    return heuristics


def run_corruption_repair_experiment():
    """
    Main experiment: Train on corrupted test set, evaluate on full corrupted set.
    """
    print("=" * 80)
    print("CORRUPTION REPAIR EXPERIMENT")
    print("Activation-Based vs Raw Repair Heuristics")
    print("=" * 80)

    # Experiment configuration
    repair_set_path = "data/edit_sets/king_penguin_test_fixed_edit_dataset.pt"
    eval_set_path = "data/edit_sets/king_penguin_corrupted_fixed_edit_dataset.pt"

    # Create experiment with reasonable bounds for comparison
    experiment = CorruptionRepairExperiment(param_change_bound=15.0, margin=15.0)

    # Create heuristics
    heuristics = create_comparison_heuristics()

    print(f"\nRepair set: {repair_set_path}")
    print(f"Evaluation set: {eval_set_path}")
    print(f"Testing {len(heuristics)} heuristics")

    # Load datasets
    repair_set = experiment.load_edit_set(repair_set_path)
    eval_set = experiment.load_edit_set(eval_set_path)

    print(f"\nRepair set: {len(repair_set['images'])} images")
    print(f"Evaluation set: {len(eval_set['images'])} images")

    all_results = []

    # Run repair experiment on small set
    print(f"\n" + "=" * 50)
    print("PHASE 1: BATCH REPAIR EXPERIMENTS")
    print("=" * 50)

    start_time = time.time()
    repair_results = experiment.run_batch_repair_experiment(
        heuristics=heuristics,
        edit_set_path=repair_set_path,
    )

    elapsed = time.time() - start_time
    print(f"Completed batch repair experiments in {elapsed:.1f}s")

    # Add evaluation phase
    print(f"\n" + "=" * 50)
    print("PHASE 2: EVALUATION ON FULL CORRUPTED SET")
    print("=" * 50)

    evaluation_results = experiment.evaluate_repaired_models(
        repair_results, eval_set_path
    )

    # Analyze results
    print(f"\n" + "=" * 50)
    print("REPAIR SUCCESS ANALYSIS")
    print("=" * 50)

    analyze_repair_results(repair_results)

    # Analyze evaluation results
    if evaluation_results:
        print(f"\n" + "=" * 50)
        print("EVALUATION RESULTS ANALYSIS")
        print("=" * 50)
        analyze_evaluation_results(evaluation_results)

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"corruption_repair_experiment_{timestamp}.json"

    experiment_data = {
        "experiment_type": "corruption_repair_comparison",
        "repair_set_path": repair_set_path,
        "eval_set_path": eval_set_path,
        "param_change_bound": experiment.param_change_bound,
        "margin": experiment.margin,
        "timestamp": timestamp,
        "repair_results": repair_results,
        "evaluation_results": evaluation_results,
        "summary": generate_summary(repair_results),
    }

    with open(filename, "w") as f:
        json.dump(experiment_data, f, indent=2)

    print(f"\nResults saved to {filename}")
    return repair_results


def analyze_repair_results(results: List[Dict[str, Any]]):
    """Analyze and display batch repair experiment results"""

    # Group by heuristic type
    activation_based = []
    raw_repair = []
    baseline = []

    for result in results:
        heuristic_name = result.get("heuristic", "")
        if "Activation" in heuristic_name:
            activation_based.append(result)
        elif "Raw" in heuristic_name:
            raw_repair.append(result)
        else:
            baseline.append(result)

    print(f"\nActivation-Based Heuristics ({len(activation_based)} heuristics tested):")
    analyze_batch_group(activation_based)

    print(f"\nRaw Repair Heuristics ({len(raw_repair)} heuristics tested):")
    analyze_batch_group(raw_repair)

    print(f"\nBaseline Heuristics ({len(baseline)} heuristics tested):")
    analyze_batch_group(baseline)

    # Overall comparison
    print(f"\n" + "=" * 30)
    print("OVERALL COMPARISON")
    print("=" * 30)

    groups = {
        "Activation-Based": activation_based,
        "Raw Repair": raw_repair,
        "Baseline": baseline,
    }

    for group_name, group_results in groups.items():
        if group_results:
            # Calculate average success rate across all heuristics in this group
            total_correct = 0
            total_attempts = 0

            for result in group_results:
                if result.get("success", False) and result.get("repair_correct"):
                    total_correct += sum(result["repair_correct"])
                    total_attempts += result.get("batch_size", 0)

            avg_layers = sum(r.get("num_layers", 0) for r in group_results) / len(
                group_results
            )
            success_rate = total_correct / total_attempts if total_attempts > 0 else 0

            print(
                f"{group_name}: {success_rate:.1%} overall success rate, {avg_layers:.1f} avg layers"
            )


def analyze_batch_group(group_results: List[Dict[str, Any]]):
    """Analyze results for a specific group of batch repair heuristics"""
    if not group_results:
        print("  No results")
        return

    successful_heuristics = []

    for result in group_results:
        if result.get("success", False):
            correct_repairs = sum(result.get("repair_correct", []))
            total_images = result.get("batch_size", 0)
            success_rate = correct_repairs / total_images if total_images > 0 else 0

            successful_heuristics.append(
                {
                    "name": result.get("heuristic", "unknown"),
                    "success_rate": success_rate,
                    "correct_repairs": correct_repairs,
                    "total_images": total_images,
                    "num_layers": result.get("num_layers", 0),
                    "param_bound": result.get("param_bound_used", 0),
                }
            )

    if successful_heuristics:
        print(
            f"  Successful heuristics: {len(successful_heuristics)}/{len(group_results)}"
        )

        # Sort by success rate
        successful_heuristics.sort(key=lambda x: x["success_rate"], reverse=True)

        for h in successful_heuristics:
            print(
                f"    {h['name']}: {h['success_rate']:.1%} ({h['correct_repairs']}/{h['total_images']} images)"
            )
            print(f"      Layers: {h['num_layers']}, Param bound: ±{h['param_bound']}")
    else:
        print(f"  No successful heuristics out of {len(group_results)} tested")


def generate_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate summary statistics for batch repair results"""

    total_heuristics = len(results)
    successful_heuristics = sum(1 for r in results if r.get("success", False))

    # Calculate overall repair statistics
    total_images_repaired = 0
    total_images_attempted = 0

    for result in results:
        if result.get("success", False) and result.get("repair_correct"):
            total_images_repaired += sum(result["repair_correct"])
            total_images_attempted += result.get("batch_size", 0)

    best_heuristic = None
    best_success_rate = 0

    # Find best performing heuristic
    for result in results:
        if result.get("success", False) and result.get("repair_correct"):
            correct_repairs = sum(result["repair_correct"])
            total_images = result.get("batch_size", 0)
            success_rate = correct_repairs / total_images if total_images > 0 else 0

            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_heuristic = {
                    "name": result.get("heuristic", "unknown"),
                    "success_rate": success_rate,
                    "correct_repairs": correct_repairs,
                    "total_images": total_images,
                    "num_layers": result.get("num_layers", 0),
                    "param_bound": result.get("param_bound_used", 0),
                }

    return {
        "total_heuristics_tested": total_heuristics,
        "successful_heuristics": successful_heuristics,
        "heuristic_success_rate": successful_heuristics / total_heuristics
        if total_heuristics > 0
        else 0,
        "total_images_attempted": total_images_attempted,
        "total_images_repaired": total_images_repaired,
        "overall_image_repair_rate": total_images_repaired / total_images_attempted
        if total_images_attempted > 0
        else 0,
        "best_heuristic": best_heuristic,
        "avg_layers_repaired": sum(r.get("num_layers", 0) for r in results)
        / total_heuristics
        if total_heuristics > 0
        else 0,
    }


def analyze_evaluation_results(eval_results: List[Dict[str, Any]]):
    """Analyze and display evaluation results"""
    if not eval_results:
        print("  No evaluation results available")
        return

    print(f"Evaluated {len(eval_results)} successfully repaired models")

    # Sort by evaluation accuracy
    eval_results.sort(key=lambda x: x["eval_accuracy"], reverse=True)

    print(f"\nTop performing heuristics on evaluation set:")
    for i, result in enumerate(eval_results[:5]):  # Top 5
        print(f"  {i + 1}. {result['heuristic']}")
        print(
            f"     Eval accuracy: {result['eval_accuracy']:.1%} ({result['eval_correct']}/{result['eval_total']})"
        )
        print(f"     Repair accuracy: {result['repair_accuracy']:.1%}")
        print(f"     Generalization: {result['generalization_score']:.2f}x")
        print(
            f"     Layers: {result['num_layers']}, Param bound: ±{result['param_bound']}"
        )

    # Calculate overall statistics
    avg_eval_accuracy = sum(r["eval_accuracy"] for r in eval_results) / len(
        eval_results
    )
    avg_repair_accuracy = sum(r["repair_accuracy"] for r in eval_results) / len(
        eval_results
    )
    avg_generalization = sum(r["generalization_score"] for r in eval_results) / len(
        eval_results
    )

    print(f"\nOverall Statistics:")
    print(f"  Average repair accuracy: {avg_repair_accuracy:.1%}")
    print(f"  Average eval accuracy: {avg_eval_accuracy:.1%}")
    print(f"  Average generalization: {avg_generalization:.2f}x")

    # Analyze by heuristic type
    activation_evals = [r for r in eval_results if "Activation" in r["heuristic"]]
    raw_repair_evals = [r for r in eval_results if "Raw" in r["heuristic"]]
    baseline_evals = [
        r
        for r in eval_results
        if "Activation" not in r["heuristic"] and "Raw" not in r["heuristic"]
    ]

    print(f"\nBy Heuristic Type:")
    for name, group in [
        ("Activation-Based", activation_evals),
        ("Raw Repair", raw_repair_evals),
        ("Baseline", baseline_evals),
    ]:
        if group:
            avg_acc = sum(r["eval_accuracy"] for r in group) / len(group)
            avg_gen = sum(r["generalization_score"] for r in group) / len(group)
            print(
                f"  {name}: {avg_acc:.1%} eval accuracy, {avg_gen:.2f}x generalization"
            )


def main():
    """Run the corruption repair experiment"""
    print("Starting Corruption Repair Experiment...")
    print("Comparing Activation-Based vs Raw Repair approaches")

    try:
        results = run_corruption_repair_experiment()
        print("\nExperiment completed successfully!")

    except Exception as e:
        print(f"\nError running experiment: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
