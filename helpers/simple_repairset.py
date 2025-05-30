import json
import torch
import os
from helpers.dataset import imagenet_mini
from helpers.models import squeezenet
from sytorch import nn

"""
Creates a repairset from misclassified images
"""


def create_repairset(
    model: nn.Module,
    max_misclassified: int = 5,
    output_dir: str = "repairset",
):
    # create output directory
    os.makedirs(output_dir, exist_ok=True)

    dataset, dataloader = imagenet_mini()

    # Storage for edit dataset
    edit_images = []
    edit_labels = []
    edit_metadata = []

    misclassified_count = 0

    print(f"Creating edit dataset with {max_misclassified} misclassified images...")
    print("-" * 60)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(model.device)
            labels = labels.to(model.device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Check for misclassification
            is_correct = predicted == labels

            # Handle misclassified images
            for i in range(images.size(0)):
                if not is_correct[i] and (
                    max_misclassified == -1 or misclassified_count < max_misclassified
                ):
                    # Store the image tensor (normalized)
                    edit_images.append(images[i].cpu())
                    # Store the correct label
                    edit_labels.append(labels[i].cpu())

                    # Store metadata
                    batch_size = dataloader.batch_size or 1
                    true_label_idx = int(labels[i].item())
                    pred_label_idx = int(predicted[i].item())
                    metadata = {
                        "image_idx": (batch_idx * batch_size) + i,
                        "true_label": true_label_idx,
                        "predicted_label": pred_label_idx,
                        "true_class": dataset.classes[true_label_idx],
                        "predicted_class": dataset.classes[pred_label_idx],
                    }
                    edit_metadata.append(metadata)

                    print(f"Added misclassified image {misclassified_count + 1}:")
                    print(
                        f"  True: {metadata['true_class']} (label {metadata['true_label']})"
                    )
                    print(
                        f"  Predicted: {metadata['predicted_class']} (label {metadata['predicted_label']})"
                    )
                    print()

                    misclassified_count += 1
                    if (
                        max_misclassified != -1
                        and misclassified_count >= max_misclassified
                    ):
                        break

            # Exit early if we've found enough misclassified images
            if max_misclassified != -1 and misclassified_count >= max_misclassified:
                break

    # Convert to tensors
    edit_images_tensor = torch.stack(edit_images)
    edit_labels_tensor = torch.stack(edit_labels)

    # Save the edit dataset
    edit_dataset_path = os.path.join(output_dir, "squeezenet_edit_dataset.pt")
    metadata_path = os.path.join(output_dir, "squeezenet_edit_metadata.json")

    torch.save(
        {
            "images": edit_images_tensor,
            "labels": edit_labels_tensor,
            "metadata": edit_metadata,
        },
        edit_dataset_path,
    )

    # Save metadata as JSON for easy inspection
    with open(metadata_path, "w") as f:
        json.dump(edit_metadata, f, indent=2)

    print("-" * 60)
    print("Edit dataset created successfully!")
    print(f"  Dataset saved to: {edit_dataset_path}")
    print(f"  Metadata saved to: {metadata_path}")
    print(f"  Number of edit examples: {len(edit_images)}")
    print(f"  Image tensor shape: {edit_images_tensor.shape}")
    print(f"  Labels tensor shape: {edit_labels_tensor.shape}")

    return edit_images_tensor, edit_labels_tensor, edit_metadata
