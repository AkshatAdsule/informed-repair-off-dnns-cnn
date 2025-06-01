#!/usr/bin/env python3
"""
Generate King Penguin Edit Set - Creates an edit set containing all king penguin images
from the ImageNet-mini validation set with their actual SqueezeNet predictions.
"""

import os
import json
import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

# Import SqueezeNet model from helpers
import sys

sys.path.append("helpers")
from models import squeezenet


def load_squeezenet_model():
    """Load the SqueezeNet model using the helpers function"""
    print("Loading SqueezeNet model from torchvision...")
    model = squeezenet(pretrained=True, eval=True)
    print("✓ SqueezeNet model loaded successfully")

    return model


def create_king_penguin_edit_set():
    """Create an edit set containing all king penguin images with their actual predictions"""

    # King penguin class information
    KING_PENGUIN_CLASS_ID = 145  # from labels.json
    KING_PENGUIN_WORDNET_ID = "n02056570"
    KING_PENGUIN_NAME = "king_penguin"

    print(
        f"Creating edit set for all {KING_PENGUIN_NAME} images (class {KING_PENGUIN_CLASS_ID}, {KING_PENGUIN_WORDNET_ID})"
    )

    # Setup paths
    val_dir = "data/imagenet-mini/val"
    output_dir = "data/edit_sets"
    os.makedirs(output_dir, exist_ok=True)

    # Define transforms (same as used in SqueezeNet training)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load validation dataset
    print("Loading validation dataset...")
    dataset = ImageFolder(val_dir, transform=transform)

    # Load SqueezeNet model for inference
    model = load_squeezenet_model()

    # Find all king penguin images
    king_penguin_indices = []
    for idx, (image_path, class_idx) in enumerate(dataset.samples):
        if class_idx == KING_PENGUIN_CLASS_ID:
            king_penguin_indices.append(idx)

    print(f"Found {len(king_penguin_indices)} king penguin images")

    if len(king_penguin_indices) == 0:
        print("No king penguin images found in validation set!")
        return

    # Collect images and metadata
    images = []
    metadata = []

    print("Processing king penguin images and running inference...")

    # Process images in batches for efficiency
    batch_size = 8
    for i in range(0, len(king_penguin_indices), batch_size):
        batch_indices = king_penguin_indices[i : i + batch_size]
        batch_images = []
        batch_metadata = []

        # Load batch of images
        for idx in batch_indices:
            image, class_idx = dataset[idx]
            batch_images.append(image)

        # Stack into batch tensor
        batch_tensor = torch.stack(batch_images)

        # Run inference
        with torch.no_grad():
            outputs = model(batch_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidences, predicted_classes = torch.max(probabilities, 1)

        # Process results
        for j, idx in enumerate(batch_indices):
            images.append(batch_images[j])

            predicted_class = predicted_classes[j].item()
            confidence = confidences[j].item()

            # Map predicted class back to wordnet ID
            # Note: This assumes the model outputs correspond to the ImageNet-mini class indices
            # which should match the validation dataset class indices
            predicted_wordnet_id = (
                dataset.classes[predicted_class]
                if predicted_class < len(dataset.classes)
                else f"class_{predicted_class}"
            )

            # Create metadata entry
            metadata_entry = {
                "image_idx": idx,
                "true_label": KING_PENGUIN_CLASS_ID,
                "predicted_label": predicted_class,
                "confidence": float(confidence),
                "true_class": KING_PENGUIN_WORDNET_ID,
                "predicted_class": predicted_wordnet_id,
                "type": "king_penguin_with_inference",
                "image_path": dataset.samples[idx][0],
                "is_correct": predicted_class == KING_PENGUIN_CLASS_ID,
            }
            metadata.append(metadata_entry)

        print(
            f"Processed {min(i + batch_size, len(king_penguin_indices))}/{len(king_penguin_indices)} images..."
        )

    # Convert images to tensor
    images_tensor = torch.stack(images)

    # Calculate statistics
    correct_predictions = sum(1 for m in metadata if m["is_correct"])
    accuracy = correct_predictions / len(metadata) if metadata else 0.0

    # Create edit set data structure
    edit_set_data = {
        "images": images_tensor,
        "metadata": {
            "class_name": KING_PENGUIN_NAME,
            "class_id": KING_PENGUIN_CLASS_ID,
            "wordnet_id": KING_PENGUIN_WORDNET_ID,
            "num_images": len(images),
            "correct_predictions": correct_predictions,
            "accuracy": accuracy,
            "source": "imagenet-mini validation set",
            "description": f"All king penguin images from validation set with SqueezeNet predictions (accuracy: {accuracy:.1%})",
        },
    }

    # Save files
    dataset_filename = f"king_penguin_with_predictions_dataset.pt"
    metadata_filename = f"king_penguin_with_predictions_metadata.json"

    dataset_path = os.path.join(output_dir, dataset_filename)
    metadata_path = os.path.join(output_dir, metadata_filename)

    print(f"Saving dataset to {dataset_path}")
    torch.save(edit_set_data, dataset_path)

    print(f"Saving metadata to {metadata_path}")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSuccessfully created king penguin edit set with predictions:")
    print(f"  - {len(images)} total king penguin images")
    print(f"  - {correct_predictions} correctly classified ({accuracy:.1%})")
    print(f"  - {len(images) - correct_predictions} misclassified ({1 - accuracy:.1%})")
    print(f"  - Dataset file: {dataset_path}")
    print(f"  - Metadata file: {metadata_path}")
    print(f"  - Total size: {os.path.getsize(dataset_path) / (1024 * 1024):.1f} MB")

    return dataset_path, metadata_path


if __name__ == "__main__":
    create_king_penguin_edit_set()
