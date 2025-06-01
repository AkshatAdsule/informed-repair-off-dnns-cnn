#!/usr/bin/env python3
"""
Generate Corrupted Edit Set - Creates corrupted versions of existing edit sets
using the imagecorruptions library to test model robustness.
"""

import os
import json
import torch
import numpy as np
import argparse
import sys
from PIL import Image
from torchvision import transforms
from imagecorruptions import corrupt, get_corruption_names

# Import SqueezeNet model from helpers
sys.path.append("helpers")
from models import squeezenet


def load_squeezenet_model():
    """Load the SqueezeNet model using the helpers function"""
    print("Loading SqueezeNet model from torchvision...")
    model = squeezenet(pretrained=True, eval=True)
    print("✓ SqueezeNet model loaded successfully")
    return model


def tensor_to_pil(tensor):
    """Convert a normalized tensor to PIL Image"""
    # Denormalize the tensor
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    denormalized = tensor * std + mean
    denormalized = torch.clamp(denormalized, 0, 1)

    # Convert to numpy and then PIL
    np_image = denormalized.permute(1, 2, 0).numpy()
    np_image = (np_image * 255).astype(np.uint8)

    return Image.fromarray(np_image)


def pil_to_tensor(pil_image):
    """Convert PIL Image to normalized tensor"""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(pil_image)


def create_corrupted_edit_set(
    source_edit_set_path,
    source_metadata_path,
    corruption_types=None,
    severity_levels=None,
    output_name=None,
    test_model_predictions=True,
):
    """
    Create corrupted versions of an existing edit set

    Args:
        source_edit_set_path (str): Path to the source edit set .pt file
        source_metadata_path (str): Path to the source metadata .json file
        corruption_types (list): List of corruption types to apply (default: common corruptions)
        severity_levels (list): List of severity levels 1-5 (default: [1, 3, 5])
        output_name (str): Custom name for output files (optional)
        test_model_predictions (bool): Whether to test model predictions on corrupted images
    """

    print(f"Loading source edit set from {source_edit_set_path}")

    # Load source edit set
    edit_data = torch.load(source_edit_set_path, map_location="cpu")

    # Handle different edit set formats
    if isinstance(edit_data, dict):
        if "images" in edit_data:
            source_images = edit_data["images"]
            source_labels = edit_data.get("labels", None)
        else:
            # Legacy format
            source_images = edit_data
            source_labels = None
    else:
        source_images = edit_data
        source_labels = None

    # Load metadata
    with open(source_metadata_path, "r") as f:
        source_metadata = json.load(f)

    print(f"✓ Loaded {len(source_images)} images from source edit set")

    # Set defaults
    if corruption_types is None:
        corruption_types = get_corruption_names(
            "common"
        )  # Get the 15 common corruptions
    if severity_levels is None:
        severity_levels = [1, 3, 5]  # Light, medium, heavy corruption

    print(
        f"Applying {len(corruption_types)} corruption types with severity levels {severity_levels}"
    )
    print(f"Corruption types: {corruption_types}")

    # Load model if we're testing predictions
    model = None
    if test_model_predictions:
        model = load_squeezenet_model()

    # Storage for corrupted edit set
    corrupted_images = []
    corrupted_metadata = []

    output_dir = "data/edit_sets"
    os.makedirs(output_dir, exist_ok=True)

    total_combinations = (
        len(source_images) * len(corruption_types) * len(severity_levels)
    )
    processed = 0

    print(f"\nProcessing {total_combinations} image-corruption combinations...")
    print("-" * 80)

    for img_idx, (source_tensor, original_meta) in enumerate(
        zip(source_images, source_metadata)
    ):
        for corruption_name in corruption_types:
            for severity in severity_levels:
                try:
                    # Convert tensor to PIL for corruption
                    pil_image = tensor_to_pil(source_tensor)

                    # Apply corruption
                    corrupted_pil = corrupt(
                        np.array(pil_image),
                        corruption_name=corruption_name,
                        severity=severity,
                    )

                    # Convert back to tensor
                    corrupted_tensor = pil_to_tensor(Image.fromarray(corrupted_pil))
                    corrupted_images.append(corrupted_tensor)

                    # Create metadata for corrupted image
                    corrupted_meta = original_meta.copy()

                    # Add corruption information
                    corrupted_meta.update(
                        {
                            "corruption_type": corruption_name,
                            "corruption_severity": severity,
                            "source_image_idx": img_idx,
                            "original_predicted_label": original_meta[
                                "predicted_label"
                            ],
                            "original_predicted_class": original_meta[
                                "predicted_class"
                            ],
                            "original_confidence": original_meta.get(
                                "confidence", None
                            ),
                            "original_is_correct": original_meta.get(
                                "is_correct", None
                            ),
                            "type": f"corrupted_{original_meta.get('type', 'unknown')}",
                        }
                    )

                    # Test model prediction on corrupted image if requested
                    if model is not None:
                        with torch.no_grad():
                            output = model(corrupted_tensor.unsqueeze(0))
                            probabilities = torch.softmax(output, dim=1)
                            confidence, predicted_label = torch.max(probabilities, 1)

                            # Update the standard prediction fields to reflect corrupted image predictions
                            predicted_label_int = int(predicted_label.item())
                            corrupted_meta.update(
                                {
                                    "predicted_label": predicted_label_int,  # Standard field
                                    "confidence": float(
                                        confidence.item()
                                    ),  # Standard field
                                    "is_correct": predicted_label_int
                                    == corrupted_meta["true_label"],  # Standard field
                                    "prediction_changed": predicted_label_int
                                    != original_meta["predicted_label"],
                                }
                            )

                            # Update predicted_class if we have access to class names
                            # Note: This assumes the dataset classes are available
                            try:
                                from torchvision.datasets import ImageFolder
                                from torchvision import transforms

                                val_dir = "data/imagenet-mini/val"
                                if os.path.exists(val_dir):
                                    temp_transform = transforms.Compose(
                                        [
                                            transforms.Resize((224, 224)),
                                            transforms.ToTensor(),
                                        ]
                                    )
                                    temp_dataset = ImageFolder(
                                        val_dir, transform=temp_transform
                                    )
                                    if predicted_label_int < len(temp_dataset.classes):
                                        corrupted_meta["predicted_class"] = (
                                            temp_dataset.classes[predicted_label_int]
                                        )
                                    else:
                                        corrupted_meta["predicted_class"] = (
                                            f"class_{predicted_label_int}"
                                        )
                            except:
                                corrupted_meta["predicted_class"] = (
                                    f"class_{predicted_label_int}"
                                )

                    corrupted_metadata.append(corrupted_meta)
                    processed += 1

                    if processed % 50 == 0:
                        print(
                            f"Processed {processed}/{total_combinations} combinations..."
                        )

                except Exception as e:
                    print(
                        f"Warning: Failed to corrupt image {img_idx} with {corruption_name} severity {severity}: {e}"
                    )
                    continue

    if not corrupted_images:
        print("No corrupted images were successfully created!")
        return None, None

    print(f"\n✓ Successfully created {len(corrupted_images)} corrupted images")

    # Convert to tensors
    corrupted_images_tensor = torch.stack(corrupted_images)

    # Generate output name
    if output_name is None:
        source_name = os.path.basename(source_edit_set_path).replace("_dataset.pt", "")
        corruption_suffix = f"{len(corruption_types)}corruptions_sev{'_'.join(map(str, severity_levels))}"
        output_name = f"{source_name}_corrupted_{corruption_suffix}"

    # Create edit set data structure
    edit_set_data = {
        "images": corrupted_images_tensor,
        "labels": torch.tensor([meta["true_label"] for meta in corrupted_metadata]),
        "metadata": corrupted_metadata,
    }

    # Save files
    dataset_filename = f"{output_name}_edit_dataset.pt"
    metadata_filename = f"{output_name}_edit_metadata.json"

    dataset_path = os.path.join(output_dir, dataset_filename)
    metadata_path = os.path.join(output_dir, metadata_filename)

    print(f"Saving dataset to {dataset_path}")
    torch.save(edit_set_data, dataset_path)

    print(f"Saving metadata to {metadata_path}")
    with open(metadata_path, "w") as f:
        json.dump(corrupted_metadata, f, indent=2)

    # Calculate statistics
    total_images = len(corrupted_images)

    if model is not None:
        original_correct = sum(
            1 for meta in corrupted_metadata if meta.get("original_is_correct", False)
        )
        corrupted_correct = sum(
            1 for meta in corrupted_metadata if meta.get("is_correct", False)
        )
        predictions_changed = sum(
            1 for meta in corrupted_metadata if meta.get("prediction_changed", False)
        )

        print(f"\n" + "=" * 80)
        print("CORRUPTED EDIT SET STATISTICS")
        print("=" * 80)
        print(f"✓ Total corrupted images: {total_images}")
        print(f"✓ Source images: {len(source_images)}")
        print(f"✓ Corruption types: {len(corruption_types)}")
        print(f"✓ Severity levels: {severity_levels}")
        print(f"✓ Original accuracy: {original_correct / total_images:.1%}")
        print(f"✓ Corrupted accuracy: {corrupted_correct / total_images:.1%}")
        print(f"✓ Predictions changed: {predictions_changed / total_images:.1%}")

        # Per-corruption statistics
        print(f"\nPer-corruption accuracy:")
        for corruption in corruption_types:
            for severity in severity_levels:
                subset = [
                    m
                    for m in corrupted_metadata
                    if m["corruption_type"] == corruption
                    and m["corruption_severity"] == severity
                ]
                if subset:
                    correct = sum(1 for m in subset if m.get("is_correct", False))
                    accuracy = correct / len(subset)
                    print(
                        f"  {corruption} (sev {severity}): {accuracy:.1%} ({correct}/{len(subset)})"
                    )

    print(f"\n✓ Dataset file: {dataset_path}")
    print(f"✓ Metadata file: {metadata_path}")
    print(f"✓ Total size: {os.path.getsize(dataset_path) / (1024 * 1024):.1f} MB")

    return dataset_path, metadata_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate corrupted edit set from existing edit set"
    )

    parser.add_argument(
        "source_edit_set",
        nargs="?",  # Make it optional
        help="Path to source edit set (.pt file) or base name (e.g., 'king_penguin_with_predictions')",
    )

    parser.add_argument(
        "--corruptions",
        nargs="*",
        help="List of corruption types to apply (default: common corruptions)",
    )

    parser.add_argument(
        "--severities",
        nargs="*",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Severity levels to apply (default: [1, 3, 5])",
    )

    parser.add_argument("--output-name", help="Custom name for output files")

    parser.add_argument(
        "--no-model-test",
        action="store_true",
        help="Skip testing model predictions on corrupted images",
    )

    parser.add_argument(
        "--list-corruptions",
        action="store_true",
        help="List available corruption types",
    )

    args = parser.parse_args()

    if args.list_corruptions:
        print("Available corruption types:")
        print("\nCommon corruptions (15):")
        for i, corruption in enumerate(get_corruption_names("common"), 1):
            print(f"  {i:2d}. {corruption}")

        print("\nValidation corruptions (4):")
        for i, corruption in enumerate(get_corruption_names("validation"), 1):
            print(f"  {i:2d}. {corruption}")

        print("\nAll corruptions:")
        for i, corruption in enumerate(get_corruption_names("all"), 1):
            print(f"  {i:2d}. {corruption}")
        return

    if not args.source_edit_set:
        print("Error: source_edit_set is required when not using --list-corruptions")
        parser.print_help()
        return

    # Resolve source paths
    if args.source_edit_set.endswith(".pt"):
        source_edit_set_path = args.source_edit_set
        source_metadata_path = args.source_edit_set.replace(
            "_dataset.pt", "_metadata.json"
        )
    else:
        # Assume it's a base name in data/edit_sets
        source_edit_set_path = f"data/edit_sets/{args.source_edit_set}_dataset.pt"
        source_metadata_path = f"data/edit_sets/{args.source_edit_set}_metadata.json"

    if not os.path.exists(source_edit_set_path):
        print(f"Error: Source edit set not found at {source_edit_set_path}")
        print(f"Available edit sets in data/edit_sets:")
        if os.path.exists("data/edit_sets"):
            for f in os.listdir("data/edit_sets"):
                if f.endswith("_dataset.pt"):
                    print(f"  {f.replace('_dataset.pt', '')}")
        return

    if not os.path.exists(source_metadata_path):
        print(f"Error: Source metadata not found at {source_metadata_path}")
        return

    print("=" * 80)
    print("CORRUPTED EDIT SET GENERATOR")
    print("=" * 80)

    try:
        dataset_path, metadata_path = create_corrupted_edit_set(
            source_edit_set_path=source_edit_set_path,
            source_metadata_path=source_metadata_path,
            corruption_types=args.corruptions,
            severity_levels=args.severities,
            output_name=args.output_name,
            test_model_predictions=not args.no_model_test,
        )

        if dataset_path and metadata_path:
            print("\n" + "=" * 80)
            print("SUCCESS! Corrupted edit set created.")
            print("=" * 80)
            print(f"Use the visualizer: uv run python edit_set_visualizer.py")

    except Exception as e:
        print(f"\n✗ Error generating corrupted edit set: {e}")
        import traceback

        traceback.print_exc()
        print("\nPlease check that:")
        print("  - The source edit set exists and is valid")
        print("  - You have sufficient disk space")
        print("  - All dependencies are installed")


if __name__ == "__main__":
    main()
