#!/usr/bin/env python3
"""
Generate Edit Set - Create the edit set from misclassified images
This script will generate the edit set data needed for the visualizer.
"""

import os
import sys
from helpers.simple_repairset import create_repairset
from helpers.models import squeezenet


def main():
    print("=" * 60)
    print("EDIT SET GENERATOR")
    print("=" * 60)

    # Check if edit set already exists
    edit_path = "data/edit_sets/squeezenet_edit_dataset.pt"
    metadata_path = "data/edit_sets/squeezenet_edit_metadata.json"

    if os.path.exists(edit_path) and os.path.exists(metadata_path):
        print(f"Edit set already exists at {edit_path}")
        print(f"Metadata already exists at {metadata_path}")

        response = input("Do you want to regenerate? (y/N): ").strip().lower()
        if response not in ["y", "yes"]:
            print("Using existing edit set.")
            return

    # Create output directory
    os.makedirs("data/edit_sets", exist_ok=True)

    print("\nLoading SqueezeNet model...")
    try:
        model = squeezenet(pretrained=True, eval=True)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return

    print("\nGenerating edit set from all misclassified images...")
    print("This may take a few minutes...")

    try:
        # Generate edit set with all misclassified images (-1 means no limit)
        edit_images, edit_labels, edit_metadata = create_repairset(
            model=model,
            max_misclassified=-1,  # Get all misclassified images
            output_dir="data/edit_sets",
        )

        print("\n" + "=" * 60)
        print("EDIT SET GENERATION COMPLETE!")
        print("=" * 60)
        print(f"✓ Generated edit set with {len(edit_images)} misclassified images")
        print(f"✓ Saved to: {edit_path}")
        print(f"✓ Metadata saved to: {metadata_path}")
        print("\nYou can now run the visualizer with:")
        print("  python edit_set_visualizer.py")

    except Exception as e:
        print(f"\n✗ Error generating edit set: {e}")
        print("\nPlease check that:")
        print("  - The imagenet-mini dataset is in data/imagenet-mini/val/")
        print("  - You have sufficient disk space")
        print("  - All dependencies are installed")


if __name__ == "__main__":
    main()
