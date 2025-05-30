#!/usr/bin/env python3
"""
Generate Confident Wrong Repair Set - Create a repair set from confidently wrong predictions

This script generates a repair set containing images where the classifier made incorrect
predictions with high confidence. This helps focus on clear misclassifications rather
than ambiguous cases.
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
from helpers.dataset import imagenet_mini
from helpers.models import squeezenet


def create_confident_wrong_repairset(
    model,
    confidence_threshold=0.8,
    max_samples=-1,
    output_dir="data/edit_sets",
    output_prefix="confident_wrong",
):
    """
    Create a repair set from confidently wrong predictions.

    Args:
        model: The model to evaluate
        confidence_threshold: Minimum confidence for wrong predictions (default: 0.8)
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

    print(f"Creating confident wrong repair set...")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"Max samples: {max_samples if max_samples > 0 else 'unlimited'}")
    print("-" * 80)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(model.device)
            labels = labels.to(model.device)
            outputs = model(images)

            # Get probabilities using softmax
            probabilities = F.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probabilities, 1)

            # Check for misclassification
            is_correct = predicted == labels

            # Handle misclassified images with high confidence
            for i in range(images.size(0)):
                total_processed += 1

                if not is_correct[i]:
                    total_misclassified += 1
                    confidence = confidences[i].item()

                    # Only include if confidence is above threshold
                    if confidence >= confidence_threshold:
                        if max_samples == -1 or samples_collected < max_samples:
                            # Store the image tensor (normalized)
                            edit_images.append(images[i].cpu())
                            # Store the correct label
                            edit_labels.append(labels[i].cpu())

                            # Store metadata including confidence
                            batch_size = dataloader.batch_size or 1
                            true_label_idx = int(labels[i].item())
                            pred_label_idx = int(predicted[i].item())

                            metadata = {
                                "image_idx": (batch_idx * batch_size) + i,
                                "true_label": true_label_idx,
                                "predicted_label": pred_label_idx,
                                "confidence": float(confidence),
                                "true_class": dataset.classes[true_label_idx],
                                "predicted_class": dataset.classes[pred_label_idx],
                                "type": "confident_wrong",
                            }
                            edit_metadata.append(metadata)

                            print(
                                f"Added confident wrong #{samples_collected + 1} (conf: {confidence:.3f}):"
                            )
                            print(
                                f"  True: {metadata['true_class']} (label {metadata['true_label']})"
                            )
                            print(
                                f"  Predicted: {metadata['predicted_class']} (label {metadata['predicted_label']})"
                            )
                            print()

                            samples_collected += 1

                            if max_samples != -1 and samples_collected >= max_samples:
                                break

                # Progress update
                if total_processed % 1000 == 0:
                    print(
                        f"Processed {total_processed} images, found {samples_collected} confident wrong predictions..."
                    )

            # Exit early if we've found enough samples
            if max_samples != -1 and samples_collected >= max_samples:
                break

    if not edit_images:
        print(
            f"No confident wrong predictions found with confidence >= {confidence_threshold}"
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
    print("CONFIDENT WRONG REPAIR SET CREATED!")
    print("-" * 80)
    print(f"✓ Dataset saved to: {dataset_path}")
    print(f"✓ Metadata saved to: {metadata_path}")
    print(f"✓ Total images processed: {total_processed}")
    print(f"✓ Total misclassified: {total_misclassified}")
    print(f"✓ Confident wrong samples: {len(edit_images)}")
    print(f"✓ Confidence threshold: {confidence_threshold}")
    print(f"✓ Image tensor shape: {edit_images_tensor.shape}")
    print(f"✓ Labels tensor shape: {edit_labels_tensor.shape}")

    # Calculate and display statistics
    confidences = [meta["confidence"] for meta in edit_metadata]
    if confidences:
        print(f"✓ Average confidence: {sum(confidences) / len(confidences):.3f}")
        print(f"✓ Min confidence: {min(confidences):.3f}")
        print(f"✓ Max confidence: {max(confidences):.3f}")

    return edit_images_tensor, edit_labels_tensor, edit_metadata


def main():
    print("=" * 80)
    print("CONFIDENT WRONG REPAIR SET GENERATOR")
    print("=" * 80)

    # Parse command line arguments
    confidence_threshold = 0.8
    max_samples = -1

    if len(sys.argv) > 1:
        try:
            confidence_threshold = float(sys.argv[1])
        except ValueError:
            print(f"Invalid confidence threshold: {sys.argv[1]}")
            print(
                "Usage: python generate_confident_wrong_repairset.py [confidence_threshold] [max_samples]"
            )
            return

    if len(sys.argv) > 2:
        try:
            max_samples = int(sys.argv[2])
        except ValueError:
            print(f"Invalid max samples: {sys.argv[2]}")
            print(
                "Usage: python generate_confident_wrong_repairset.py [confidence_threshold] [max_samples]"
            )
            return

    # Check if repair set already exists
    output_prefix = f"confident_wrong_{confidence_threshold:.2f}"
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

    print(f"\nGenerating confident wrong repair set...")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"Max samples: {max_samples if max_samples > 0 else 'unlimited'}")

    try:
        edit_images, edit_labels, edit_metadata = create_confident_wrong_repairset(
            model=model,
            confidence_threshold=confidence_threshold,
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
