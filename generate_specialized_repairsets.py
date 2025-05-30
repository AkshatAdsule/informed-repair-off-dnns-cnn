#!/usr/bin/env python3
"""
Generate Specialized Repair Sets - Create targeted repair sets based on prediction characteristics

This script provides utilities to generate specialized repair sets:
1. "Confident Wrong" - high confidence incorrect predictions (avoiding ambiguous cases)
2. "Closest" - misclassifications that were almost correct (small margin between predicted and true)

Usage:
    python generate_specialized_repairsets.py confident_wrong [confidence_threshold] [max_samples]
    python generate_specialized_repairsets.py closest [min_true_prob] [max_margin] [max_samples]
    python generate_specialized_repairsets.py both [options...]
    python generate_specialized_repairsets.py analyze
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
    verbose=True,
):
    """Create a repair set from confidently wrong predictions."""
    os.makedirs(output_dir, exist_ok=True)

    dataset, dataloader = imagenet_mini()
    edit_images, edit_labels, edit_metadata = [], [], []
    samples_collected = total_processed = total_misclassified = 0

    if verbose:
        print(
            f"Creating confident wrong repair set (threshold: {confidence_threshold})..."
        )
        print("-" * 60)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(model.device)
            labels = labels.to(model.device)
            outputs = model(images)

            probabilities = F.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probabilities, 1)
            is_correct = predicted == labels

            for i in range(images.size(0)):
                total_processed += 1

                if not is_correct[i]:
                    total_misclassified += 1
                    confidence = confidences[i].item()

                    if confidence >= confidence_threshold:
                        if max_samples == -1 or samples_collected < max_samples:
                            edit_images.append(images[i].cpu())
                            edit_labels.append(labels[i].cpu())

                            batch_size = dataloader.batch_size or 1
                            metadata = {
                                "image_idx": (batch_idx * batch_size) + i,
                                "true_label": int(labels[i].item()),
                                "predicted_label": int(predicted[i].item()),
                                "confidence": float(confidence),
                                "true_class": dataset.classes[labels[i]],
                                "predicted_class": dataset.classes[predicted[i]],
                                "type": "confident_wrong",
                            }
                            edit_metadata.append(metadata)

                            if verbose and samples_collected < 10:  # Show first few
                                print(
                                    f"  #{samples_collected + 1}: {metadata['true_class']} → {metadata['predicted_class']} (conf: {confidence:.3f})"
                                )

                            samples_collected += 1
                            if max_samples != -1 and samples_collected >= max_samples:
                                break

                if total_processed % 1000 == 0 and verbose:
                    print(
                        f"  Processed {total_processed}, found {samples_collected} confident wrong..."
                    )

            if max_samples != -1 and samples_collected >= max_samples:
                break

    if not edit_images:
        if verbose:
            print(
                f"No confident wrong predictions found with confidence >= {confidence_threshold}"
            )
        return (
            None,
            None,
            None,
            {
                "total_processed": total_processed,
                "total_misclassified": total_misclassified,
            },
        )

    # Save results
    output_prefix = f"confident_wrong_{confidence_threshold:.2f}"
    edit_images_tensor = torch.stack(edit_images)
    edit_labels_tensor = torch.stack(edit_labels)

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
    with open(metadata_path, "w") as f:
        json.dump(edit_metadata, f, indent=2)

    stats = {
        "total_processed": total_processed,
        "total_misclassified": total_misclassified,
        "samples_collected": samples_collected,
        "dataset_path": dataset_path,
        "metadata_path": metadata_path,
        "avg_confidence": sum(m["confidence"] for m in edit_metadata)
        / len(edit_metadata),
        "min_confidence": min(m["confidence"] for m in edit_metadata),
        "max_confidence": max(m["confidence"] for m in edit_metadata),
    }

    if verbose:
        print(f"✓ Confident wrong repair set created: {samples_collected} samples")
        print(f"  Average confidence: {stats['avg_confidence']:.3f}")

    return edit_images_tensor, edit_labels_tensor, edit_metadata, stats


def create_closest_repairset(
    model,
    min_true_prob=0.1,
    max_margin=0.3,
    max_samples=-1,
    output_dir="data/edit_sets",
    verbose=True,
):
    """Create a repair set from closest misclassifications."""
    os.makedirs(output_dir, exist_ok=True)

    dataset, dataloader = imagenet_mini()
    edit_images, edit_labels, edit_metadata = [], [], []
    samples_collected = total_processed = total_misclassified = 0

    if verbose:
        print(
            f"Creating closest repair set (min_prob: {min_true_prob}, max_margin: {max_margin})..."
        )
        print("-" * 60)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(model.device)
            labels = labels.to(model.device)
            outputs = model(images)

            probabilities = F.softmax(outputs, dim=1)
            predicted_probs, predicted = torch.max(probabilities, 1)
            is_correct = predicted == labels

            for i in range(images.size(0)):
                total_processed += 1

                if not is_correct[i]:
                    total_misclassified += 1

                    predicted_prob = predicted_probs[i].item()
                    true_label_idx = labels[i].item()
                    true_prob = probabilities[i][true_label_idx].item()
                    margin = predicted_prob - true_prob

                    if true_prob >= min_true_prob and margin <= max_margin:
                        if max_samples == -1 or samples_collected < max_samples:
                            edit_images.append(images[i].cpu())
                            edit_labels.append(labels[i].cpu())

                            batch_size = dataloader.batch_size or 1
                            metadata = {
                                "image_idx": (batch_idx * batch_size) + i,
                                "true_label": int(true_label_idx),
                                "predicted_label": int(predicted[i].item()),
                                "true_probability": float(true_prob),
                                "predicted_probability": float(predicted_prob),
                                "margin": float(margin),
                                "true_class": dataset.classes[true_label_idx],
                                "predicted_class": dataset.classes[predicted[i]],
                                "type": "closest",
                            }
                            edit_metadata.append(metadata)

                            if verbose and samples_collected < 10:  # Show first few
                                print(
                                    f"  #{samples_collected + 1}: {metadata['true_class']} → {metadata['predicted_class']} (margin: {margin:.3f})"
                                )

                            samples_collected += 1
                            if max_samples != -1 and samples_collected >= max_samples:
                                break

                if total_processed % 1000 == 0 and verbose:
                    print(
                        f"  Processed {total_processed}, found {samples_collected} closest..."
                    )

            if max_samples != -1 and samples_collected >= max_samples:
                break

    if not edit_images:
        if verbose:
            print(f"No closest misclassifications found with criteria")
        return (
            None,
            None,
            None,
            {
                "total_processed": total_processed,
                "total_misclassified": total_misclassified,
            },
        )

    # Save results
    output_prefix = f"closest_{min_true_prob:.2f}_{max_margin:.2f}"
    edit_images_tensor = torch.stack(edit_images)
    edit_labels_tensor = torch.stack(edit_labels)

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
    with open(metadata_path, "w") as f:
        json.dump(edit_metadata, f, indent=2)

    stats = {
        "total_processed": total_processed,
        "total_misclassified": total_misclassified,
        "samples_collected": samples_collected,
        "dataset_path": dataset_path,
        "metadata_path": metadata_path,
        "avg_true_prob": sum(m["true_probability"] for m in edit_metadata)
        / len(edit_metadata),
        "avg_margin": sum(m["margin"] for m in edit_metadata) / len(edit_metadata),
        "min_margin": min(m["margin"] for m in edit_metadata),
        "max_margin": max(m["margin"] for m in edit_metadata),
    }

    if verbose:
        print(f"✓ Closest repair set created: {samples_collected} samples")
        print(f"  Average margin: {stats['avg_margin']:.3f}")

    return edit_images_tensor, edit_labels_tensor, edit_metadata, stats


def analyze_repair_sets():
    """Analyze existing repair sets and provide statistics."""
    print("=" * 80)
    print("REPAIR SET ANALYSIS")
    print("=" * 80)

    edit_sets_dir = "data/edit_sets"
    if not os.path.exists(edit_sets_dir):
        print("No edit sets directory found.")
        return

    # Find all metadata files
    metadata_files = [
        f for f in os.listdir(edit_sets_dir) if f.endswith("_metadata.json")
    ]

    if not metadata_files:
        print("No repair set metadata files found.")
        return

    print(f"Found {len(metadata_files)} repair sets:\n")

    for metadata_file in sorted(metadata_files):
        metadata_path = os.path.join(edit_sets_dir, metadata_file)
        dataset_file = metadata_file.replace("_metadata.json", "_dataset.pt")
        dataset_path = os.path.join(edit_sets_dir, dataset_file)

        print(f"📊 {metadata_file}")
        print("-" * 60)

        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            if os.path.exists(dataset_path):
                dataset = torch.load(dataset_path)
                images_shape = dataset["images"].shape
                print(f"  Samples: {len(metadata)}")
                print(f"  Image shape: {images_shape}")
            else:
                print(f"  Samples: {len(metadata)} (dataset file missing)")

            if metadata:
                sample = metadata[0]
                repair_type = sample.get("type", "unknown")
                print(f"  Type: {repair_type}")

                if repair_type == "confident_wrong":
                    confidences = [m["confidence"] for m in metadata]
                    print(
                        f"  Confidence range: {min(confidences):.3f} - {max(confidences):.3f}"
                    )
                    print(
                        f"  Average confidence: {sum(confidences) / len(confidences):.3f}"
                    )

                elif repair_type == "closest":
                    margins = [m["margin"] for m in metadata]
                    true_probs = [m["true_probability"] for m in metadata]
                    print(f"  Margin range: {min(margins):.3f} - {max(margins):.3f}")
                    print(f"  Average margin: {sum(margins) / len(margins):.3f}")
                    print(
                        f"  True prob range: {min(true_probs):.3f} - {max(true_probs):.3f}"
                    )

                # Class distribution
                true_classes = [m["true_class"] for m in metadata]
                pred_classes = [m["predicted_class"] for m in metadata]
                unique_true = len(set(true_classes))
                unique_pred = len(set(pred_classes))
                print(f"  Unique true classes: {unique_true}")
                print(f"  Unique predicted classes: {unique_pred}")

        except Exception as e:
            print(f"  Error analyzing: {e}")

        print()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1].lower()

    if command == "analyze":
        analyze_repair_sets()
        return

    print("=" * 80)
    print("SPECIALIZED REPAIR SET GENERATOR")
    print("=" * 80)

    # Load model
    print("Loading SqueezeNet model...")
    try:
        model = squeezenet(pretrained=True, eval=True)
        print("✓ Model loaded successfully\n")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return

    if command == "confident_wrong":
        confidence_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
        max_samples = int(sys.argv[3]) if len(sys.argv) > 3 else -1

        _, _, _, stats = create_confident_wrong_repairset(
            model, confidence_threshold, max_samples, verbose=True
        )

        if stats["samples_collected"] > 0:
            print(f"\n✓ Confident wrong repair set complete!")
            print(f"  Threshold: {confidence_threshold}")
            print(f"  Samples: {stats['samples_collected']}")
            print(f"  Dataset: {stats['dataset_path']}")

    elif command == "closest":
        min_true_prob = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
        max_margin = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
        max_samples = int(sys.argv[4]) if len(sys.argv) > 4 else -1

        _, _, _, stats = create_closest_repairset(
            model, min_true_prob, max_margin, max_samples, verbose=True
        )

        if stats["samples_collected"] > 0:
            print(f"\n✓ Closest repair set complete!")
            print(f"  Min true prob: {min_true_prob}")
            print(f"  Max margin: {max_margin}")
            print(f"  Samples: {stats['samples_collected']}")
            print(f"  Dataset: {stats['dataset_path']}")

    elif command == "both":
        print("Creating both repair sets...\n")

        # Confident wrong with default parameters
        confidence_threshold = 0.8
        print("1. Creating confident wrong repair set...")
        _, _, _, stats1 = create_confident_wrong_repairset(
            model,
            confidence_threshold,
            500,
            verbose=True,  # Limit to 500 each for demo
        )

        print("\n" + "=" * 60 + "\n")

        # Closest with default parameters
        min_true_prob, max_margin = 0.1, 0.3
        print("2. Creating closest repair set...")
        _, _, _, stats2 = create_closest_repairset(
            model, min_true_prob, max_margin, 500, verbose=True
        )

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Confident wrong: {stats1.get('samples_collected', 0)} samples")
        print(f"Closest: {stats2.get('samples_collected', 0)} samples")
        print(
            "\nRun 'python generate_specialized_repairsets.py analyze' to see detailed statistics."
        )

    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()
