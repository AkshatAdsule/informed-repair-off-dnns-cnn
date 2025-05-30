#!/usr/bin/env python3
"""
Generate Closest Repair Set - Create a repair set from "almost correct" predictions

This script generates a repair set containing images where the classifier made incorrect
predictions but the true label had relatively high probability. These are cases where
the model was "close" to getting it right.
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
from helpers.dataset import imagenet_mini
from helpers.models import squeezenet


def create_closest_repairset(
    model,
    min_true_prob=0.1,
    max_margin=0.3,
    max_samples=-1,
    output_dir="data/edit_sets",
    output_prefix="closest",
):
    """
    Create a repair set from "closest" misclassifications.

    Args:
        model: The model to evaluate
        min_true_prob: Minimum probability for the true label (default: 0.1)
        max_margin: Maximum margin between predicted and true probabilities (default: 0.3)
        max_samples: Maximum number of samples to collect (-1 for all)
        output_dir: Directory to save the repair set
        output_prefix: Prefix for output files
    """
    os.makedirs(output_dir, exist_ok=True)

    dataset, dataloader = imagenet_mini()

    # Storage for edit dataset
    edit_images = []
    edit_labels = []
    edit_metadata = []

    samples_collected = 0
    total_processed = 0
    total_misclassified = 0

    print(f"Creating closest misclassification repair set...")
    print(f"Minimum true label probability: {min_true_prob}")
    print(f"Maximum prediction margin: {max_margin}")
    print(f"Max samples: {max_samples if max_samples > 0 else 'unlimited'}")
    print("-" * 80)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(model.device)
            labels = labels.to(model.device)
            outputs = model(images)

            # Get probabilities using softmax
            probabilities = F.softmax(outputs, dim=1)
            predicted_probs, predicted = torch.max(probabilities, 1)

            # Check for misclassification
            is_correct = predicted == labels

            # Handle misclassified images that were "close"
            for i in range(images.size(0)):
                total_processed += 1

                if not is_correct[i]:
                    total_misclassified += 1

                    predicted_prob = predicted_probs[i].item()
                    true_label_idx = labels[i].item()
                    true_prob = probabilities[i][true_label_idx].item()
                    margin = predicted_prob - true_prob

                    # Check if this is a "close" misclassification
                    if true_prob >= min_true_prob and margin <= max_margin:
                        if max_samples == -1 or samples_collected < max_samples:
                            # Store the image tensor (normalized)
                            edit_images.append(images[i].cpu())
                            # Store the correct label
                            edit_labels.append(labels[i].cpu())

                            # Store metadata including probabilities and margin
                            batch_size = dataloader.batch_size or 1
                            pred_label_idx = int(predicted[i].item())

                            metadata = {
                                "image_idx": (batch_idx * batch_size) + i,
                                "true_label": int(true_label_idx),
                                "predicted_label": pred_label_idx,
                                "true_probability": float(true_prob),
                                "predicted_probability": float(predicted_prob),
                                "margin": float(margin),
                                "true_class": dataset.classes[true_label_idx],
                                "predicted_class": dataset.classes[pred_label_idx],
                                "type": "closest",
                            }
                            edit_metadata.append(metadata)

                            print(
                                f"Added closest #{samples_collected + 1} (margin: {margin:.3f}):"
                            )
                            print(
                                f"  True: {metadata['true_class']} (prob: {true_prob:.3f})"
                            )
                            print(
                                f"  Predicted: {metadata['predicted_class']} (prob: {predicted_prob:.3f})"
                            )
                            print()

                            samples_collected += 1

                            if max_samples != -1 and samples_collected >= max_samples:
                                break

                # Progress update
                if total_processed % 1000 == 0:
                    print(
                        f"Processed {total_processed} images, found {samples_collected} closest misclassifications..."
                    )

            # Exit early if we've found enough samples
            if max_samples != -1 and samples_collected >= max_samples:
                break

    if not edit_images:
        print(
            f"No closest misclassifications found with true_prob >= {min_true_prob} and margin <= {max_margin}"
        )
        return None, None, None

    # Convert to tensors
    edit_images_tensor = torch.stack(edit_images)
    edit_labels_tensor = torch.stack(edit_labels)

    # Save the edit dataset
    dataset_path = os.path.join(output_dir, f"{output_prefix}_edit_dataset.pt")
    metadata_path = os.path.join(output_dir, f"{output_prefix}_edit_metadata.json")

    torch.save(
        {
            "images": edit_images_tensor,
            "labels": edit_labels_tensor,
            "metadata": edit_metadata,
        },
        dataset_path,
    )

    # Save metadata as JSON for easy inspection
    with open(metadata_path, "w") as f:
        json.dump(edit_metadata, f, indent=2)

    print("-" * 80)
    print("CLOSEST MISCLASSIFICATION REPAIR SET CREATED!")
    print("-" * 80)
    print(f"✓ Dataset saved to: {dataset_path}")
    print(f"✓ Metadata saved to: {metadata_path}")
    print(f"✓ Total images processed: {total_processed}")
    print(f"✓ Total misclassified: {total_misclassified}")
    print(f"✓ Closest misclassifications: {len(edit_images)}")
    print(f"✓ Min true probability: {min_true_prob}")
    print(f"✓ Max margin: {max_margin}")
    print(f"✓ Image tensor shape: {edit_images_tensor.shape}")
    print(f"✓ Labels tensor shape: {edit_labels_tensor.shape}")

    # Calculate and display statistics
    true_probs = [meta["true_probability"] for meta in edit_metadata]
    margins = [meta["margin"] for meta in edit_metadata]
    if true_probs and margins:
        print(f"✓ Average true probability: {sum(true_probs) / len(true_probs):.3f}")
        print(f"✓ Average margin: {sum(margins) / len(margins):.3f}")
        print(f"✓ Min margin: {min(margins):.3f}")
        print(f"✓ Max margin: {max(margins):.3f}")

    return edit_images_tensor, edit_labels_tensor, edit_metadata


def main():
    print("=" * 80)
    print("CLOSEST MISCLASSIFICATION REPAIR SET GENERATOR")
    print("=" * 80)

    # Parse command line arguments
    min_true_prob = 0.1
    max_margin = 0.3
    max_samples = -1

    if len(sys.argv) > 1:
        try:
            min_true_prob = float(sys.argv[1])
        except ValueError:
            print(f"Invalid min_true_prob: {sys.argv[1]}")
            print(
                "Usage: python generate_closest_repairset.py [min_true_prob] [max_margin] [max_samples]"
            )
            return

    if len(sys.argv) > 2:
        try:
            max_margin = float(sys.argv[2])
        except ValueError:
            print(f"Invalid max_margin: {sys.argv[2]}")
            print(
                "Usage: python generate_closest_repairset.py [min_true_prob] [max_margin] [max_samples]"
            )
            return

    if len(sys.argv) > 3:
        try:
            max_samples = int(sys.argv[3])
        except ValueError:
            print(f"Invalid max_samples: {sys.argv[3]}")
            print(
                "Usage: python generate_closest_repairset.py [min_true_prob] [max_margin] [max_samples]"
            )
            return

    # Check if repair set already exists
    output_prefix = f"closest_{min_true_prob:.2f}_{max_margin:.2f}"
    dataset_path = f"data/edit_sets/{output_prefix}_edit_dataset.pt"
    metadata_path = f"data/edit_sets/{output_prefix}_edit_metadata.json"

    if os.path.exists(dataset_path) and os.path.exists(metadata_path):
        print(f"Repair set already exists at {dataset_path}")
        response = input("Do you want to regenerate? (y/N): ").strip().lower()
        if response not in ["y", "yes"]:
            print("Using existing repair set.")
            return

    print("\nLoading SqueezeNet model...")
    try:
        model = squeezenet(pretrained=True, eval=True)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return

    print(f"\nGenerating closest misclassification repair set...")
    print(f"Minimum true probability: {min_true_prob}")
    print(f"Maximum margin: {max_margin}")
    print(f"Max samples: {max_samples if max_samples > 0 else 'unlimited'}")

    try:
        edit_images, edit_labels, edit_metadata = create_closest_repairset(
            model=model,
            min_true_prob=min_true_prob,
            max_margin=max_margin,
            max_samples=max_samples,
            output_prefix=output_prefix,
        )

        if edit_images is not None:
            print("\n" + "=" * 80)
            print("SUCCESS! You can now use this repair set for analysis.")
            print("=" * 80)

    except Exception as e:
        print(f"\n✗ Error generating repair set: {e}")
        print("\nPlease check that:")
        print("  - The imagenet-mini dataset is in data/imagenet-mini/val/")
        print("  - You have sufficient disk space")
        print("  - All dependencies are installed")


if __name__ == "__main__":
    main()
