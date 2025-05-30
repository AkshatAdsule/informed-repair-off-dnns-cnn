import torch
import sytorch as st
import os
from typing import List, Dict, Any, Callable, Tuple
from helpers.models import squeezenet
import json
from datetime import datetime

# Import heuristics from the package
from heuristics import LayerSelectionHeuristic

# SyTorch setup
device = st.device("cpu")
dtype = st.float64


class RepairExperiment:
    """Framework for running repair experiments"""

    def __init__(self, param_change_bound: float = 15.0, margin: float = 15.0):
        self.param_change_bound = param_change_bound
        self.margin = margin
        self.results = []

    def load_model(self):
        """Load and prepare the model"""
        torch_model = squeezenet(pretrained=True, eval=True)
        return torch_model.to(dtype=dtype, device=device)

    def load_edit_set(
        self, edit_set_path: str = "data/edit_sets/squeezenet_edit_dataset.pt"
    ):
        """Load the edit set"""
        edit_data = torch.load(edit_set_path, map_location=device)
        return {
            "images": edit_data["images"].to(dtype=dtype, device=device),
            "labels": edit_data["labels"].to(device=device),
            "metadata": edit_data.get("metadata", []),
        }

    def run_single_repair(
        self, model, image, true_label, heuristic: LayerSelectionHeuristic
    ) -> Dict[str, Any]:
        """Run repair on a single image with given heuristic"""

        # Setup solver and model
        solver = st.GurobiSolver()
        N = model.deepcopy().to(solver).repair()

        # Get layers to make symbolic
        layers_to_repair = heuristic.select_layers(N)

        # Validate the selection
        if not heuristic.validate_selection(layers_to_repair):
            raise ValueError(f"Invalid layer selection from heuristic {heuristic.name}")

        # Make selected layers symbolic
        for layer_name, layer in layers_to_repair:
            if hasattr(layer, "weight") and layer.weight is not None:
                layer.weight.requires_symbolic_(
                    lb=-self.param_change_bound, ub=self.param_change_bound
                )
            if hasattr(layer, "bias") and layer.bias is not None:
                layer.bias.requires_symbolic_(
                    lb=-self.param_change_bound, ub=self.param_change_bound
                )

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

    def run_experiment(
        self,
        heuristics: List[LayerSelectionHeuristic],
        num_images: int = 5,
        edit_set_path: str = "data/edit_sets/squeezenet_edit_dataset.pt",
    ) -> List[Dict[str, Any]]:
        """Run repair experiments with different heuristics"""

        print("Loading model and edit set...")
        model = self.load_model()
        edit_set = self.load_edit_set(edit_set_path)

        # Limit to available images
        num_images = min(num_images, len(edit_set["images"]))

        results = []

        for img_idx in range(num_images):
            image = edit_set["images"][img_idx].unsqueeze(0)
            true_label = edit_set["labels"][img_idx].item()

            print(f"\nImage {img_idx + 1}/{num_images} (True label: {true_label})")

            if edit_set["metadata"] and img_idx < len(edit_set["metadata"]):
                meta = edit_set["metadata"][img_idx]
                print(f"  Class: {meta.get('true_class', 'N/A')}")

            for heuristic in heuristics:
                print(f"  Testing {heuristic.name}...")

                try:
                    result = self.run_single_repair(model, image, true_label, heuristic)
                    result["image_idx"] = img_idx
                    results.append(result)

                    status = "SUCCESS" if result["repair_correct"] else "FAILED"
                    print(
                        f"    {status}: {result['original_pred']} -> {result['repaired_pred']} (Target: {true_label})"
                    )
                    print(f"    Description: {result['heuristic_description']}")

                except Exception as e:
                    print(f"    ERROR: {e}")
                    results.append(
                        {
                            "image_idx": img_idx,
                            "heuristic": heuristic.name,
                            "success": False,
                            "error": str(e),
                        }
                    )

        return results

    def save_results(self, results: List[Dict[str, Any]], filename: str = None):
        """Save experiment results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"repair_experiments_{timestamp}.json"

        experiment_config = {
            "param_change_bound": self.param_change_bound,
            "margin": self.margin,
            "timestamp": datetime.now().isoformat(),
            "results": results,
        }

        with open(filename, "w") as f:
            json.dump(experiment_config, f, indent=2)

        print(f"Results saved to {filename}")

    def print_summary(self, results: List[Dict[str, Any]]):
        """Print summary of experiment results"""

        # Group by heuristic
        by_heuristic = {}
        for result in results:
            heuristic = result.get("heuristic", "unknown")
            if heuristic not in by_heuristic:
                by_heuristic[heuristic] = []
            by_heuristic[heuristic].append(result)

        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)

        for heuristic, heuristic_results in by_heuristic.items():
            successful_repairs = [
                r for r in heuristic_results if r.get("repair_correct", False)
            ]
            total_attempts = len(heuristic_results)

            print(f"\n{heuristic.upper()}:")

            # Print description if available
            if heuristic_results and "heuristic_description" in heuristic_results[0]:
                print(f"  Description: {heuristic_results[0]['heuristic_description']}")

            print(
                f"  Success rate: {len(successful_repairs)}/{total_attempts} ({len(successful_repairs) / total_attempts * 100:.1f}%)"
            )

            if successful_repairs:
                avg_layers = sum(
                    r.get("num_layers", 0) for r in successful_repairs
                ) / len(successful_repairs)
                avg_complexity = sum(
                    r.get("complexity_score", 0) for r in successful_repairs
                ) / len(successful_repairs)
                print(f"  Avg layers repaired: {avg_layers:.1f}")
                print(f"  Avg complexity score: {avg_complexity:.0f}")


def main():
    """Example usage of the experiment framework"""

    # Import specific heuristics
    from heuristics import (
        ClassifierOnlyHeuristic,
        LastNLayersHeuristic,
        FireModuleHeuristic,
        GradientBasedHeuristic,
        AdaptiveHeuristic,
    )

    # Define heuristics to test
    heuristics = [
        ClassifierOnlyHeuristic(),
        LastNLayersHeuristic(2),
        LastNLayersHeuristic(3),
        FireModuleHeuristic(1),
        GradientBasedHeuristic(3),
        AdaptiveHeuristic("confidence_based", max_layers=3),
    ]

    # Run experiment
    experiment = RepairExperiment(param_change_bound=15.0, margin=15.0)
    results = experiment.run_experiment(heuristics, num_images=3)

    # Print summary and save results
    experiment.print_summary(results)
    experiment.save_results(results)


if __name__ == "__main__":
    main()
