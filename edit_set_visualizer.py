#!/usr/bin/env python3
"""
Edit Set Visualizer - Web interface for exploring the edit set
containing misclassified images from SqueezeNet on ImageNet-mini.
"""

import os
import json
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

try:
    import seaborn as sns

    sns.set_style("whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from collections import Counter, defaultdict
from flask import Flask, render_template, send_file, jsonify, request
from torchvision.datasets import ImageFolder
from torchvision import transforms
import io
import base64
from PIL import Image
import tempfile

app = Flask(__name__)


class EditSetVisualizer:
    def __init__(self):
        self.edit_data = None
        self.dataset = None
        self.class_to_idx = None
        self.idx_to_class = None
        self.metadata = None
        self.current_edit_set = None
        self.available_edit_sets = self.discover_edit_sets()
        # Load class labels mapping
        self.class_labels, self.wordnet_to_name = self.load_class_labels()
        # Load default edit set
        if self.available_edit_sets:
            self.load_edit_set(list(self.available_edit_sets.keys())[0])
        self.load_dataset_info()

    def load_class_labels(self):
        """Load the ImageNet class labels from labels.json"""
        labels_path = "data/imagenet-mini/labels.json"
        try:
            with open(labels_path, "r") as f:
                labels_data = json.load(f)

            # Convert to mappings for both class_id -> name and wordnet_id -> name
            class_labels = {}  # class_id -> name
            wordnet_to_name = {}  # wordnet_id -> name

            for class_id, (wordnet_id, class_name) in labels_data.items():
                class_labels[class_id] = class_name
                wordnet_to_name[wordnet_id] = class_name

            print(f"Loaded {len(class_labels)} class labels from {labels_path}")
            return class_labels, wordnet_to_name
        except Exception as e:
            print(f"Warning: Could not load class labels from {labels_path}: {e}")
            return {}, {}

    def get_class_name(self, class_id):
        """Convert class ID to human-readable name"""
        if isinstance(class_id, (int, float)):
            class_id = str(int(class_id))
        elif isinstance(class_id, str):
            # Handle both string numbers and wordnet IDs
            if class_id.startswith("n"):
                # If it's a wordnet ID, look it up directly
                return self.wordnet_to_name.get(class_id, class_id)
            # Otherwise treat as class ID
            pass
        else:
            class_id = str(class_id)

        return self.class_labels.get(class_id, f"Class_{class_id}")

    def discover_edit_sets(self):
        """Discover all available edit sets in the data/edit_sets directory"""
        edit_sets_dir = "data/edit_sets"
        available_sets = {}

        if not os.path.exists(edit_sets_dir):
            return available_sets

        # Look for .pt files and their corresponding .json metadata files
        for filename in os.listdir(edit_sets_dir):
            if filename.endswith("_dataset.pt"):
                # Extract the base name (remove _dataset.pt)
                base_name = filename[:-11]  # Remove "_dataset.pt"
                metadata_file = f"{base_name}_metadata.json"

                dataset_path = os.path.join(edit_sets_dir, filename)
                metadata_path = os.path.join(edit_sets_dir, metadata_file)

                if os.path.exists(metadata_path):
                    # Get file sizes for display
                    dataset_size = os.path.getsize(dataset_path)
                    metadata_size = os.path.getsize(metadata_path)

                    # Create a human-readable name
                    display_name = base_name.replace("_", " ").title()

                    available_sets[base_name] = {
                        "display_name": display_name,
                        "dataset_path": dataset_path,
                        "metadata_path": metadata_path,
                        "dataset_size": dataset_size,
                        "metadata_size": metadata_size,
                    }

        return available_sets

    def load_edit_set(self, edit_set_name):
        """Load a specific edit set by name"""
        if edit_set_name not in self.available_edit_sets:
            raise ValueError(f"Edit set '{edit_set_name}' not found")

        edit_set_info = self.available_edit_sets[edit_set_name]

        # Load edit set data
        self.edit_data = torch.load(edit_set_info["dataset_path"], map_location="cpu")

        # Load metadata
        with open(edit_set_info["metadata_path"], "r") as f:
            self.metadata = json.load(f)

        self.current_edit_set = edit_set_name
        print(
            f"Loaded edit set: {edit_set_info['display_name']} ({len(self.metadata)} images)"
        )

    def load_dataset_info(self):
        """Load dataset for class information"""
        val_dir = "data/imagenet-mini/val"
        if os.path.exists(val_dir):
            # Use basic transform just to get class info
            transform = transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor()]
            )
            self.dataset = ImageFolder(val_dir, transform=transform)
            self.class_to_idx = self.dataset.class_to_idx
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def get_available_edit_sets(self):
        """Return information about available edit sets"""
        return {
            name: {
                "display_name": info["display_name"],
                "num_images": len(json.load(open(info["metadata_path"])))
                if os.path.exists(info["metadata_path"])
                else 0,
                "dataset_size_mb": round(info["dataset_size"] / (1024 * 1024), 1),
                "is_current": name == self.current_edit_set,
            }
            for name, info in self.available_edit_sets.items()
        }

    def get_image_tensor_as_base64(self, image_tensor):
        """Convert image tensor to base64 for web display"""
        # Denormalize the image (assuming ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        denorm_image = image_tensor * std + mean
        denorm_image = torch.clamp(denorm_image, 0, 1)

        # Convert to PIL Image
        pil_image = transforms.ToPILImage()(denorm_image)

        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=90)
        img_str = base64.b64encode(buffer.getvalue()).decode()

        return img_str

    def get_distribution_data(self):
        """Get distribution statistics for the edit set"""
        if not self.metadata:
            return {}

        # Count by true class
        true_class_counts = Counter()
        pred_class_counts = Counter()
        confusion_pairs = Counter()

        for item in self.metadata:
            true_class = item["true_class"]
            pred_class = item["predicted_class"]

            # Convert to human-readable names
            true_class_name = self.get_class_name(true_class)
            pred_class_name = self.get_class_name(pred_class)

            true_class_counts[true_class_name] += 1
            pred_class_counts[pred_class_name] += 1
            confusion_pairs[(true_class_name, pred_class_name)] += 1

        # Convert tuple keys to strings for JSON serialization
        confusion_pairs_serializable = {
            f"{true_cls} → {pred_cls}": count
            for (true_cls, pred_cls), count in confusion_pairs.items()
        }

        return {
            "true_class_counts": dict(true_class_counts),
            "pred_class_counts": dict(pred_class_counts),
            "confusion_pairs": confusion_pairs_serializable,
            "total_images": len(self.metadata),
        }

    def create_distribution_plot(self, plot_type="true_classes"):
        """Create distribution plots as base64 images"""
        try:
            dist_data = self.get_distribution_data()

            # Clear any existing plots
            plt.clf()
            plt.close("all")

            # Create new figure with adequate space
            fig, ax = plt.subplots(figsize=(14, 9))

            if plot_type == "true_classes":
                data = dist_data["true_class_counts"]
                title = "Distribution of True Classes in Edit Set"
                xlabel = "True Class"
            elif plot_type == "predicted_classes":
                data = dist_data["pred_class_counts"]
                title = "Distribution of Predicted Classes in Edit Set"
                xlabel = "Predicted Class"
            else:
                data = {}
                title = "Unknown Plot Type"
                xlabel = "Class"

            if not data:
                ax.text(
                    0.5,
                    0.5,
                    "No data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(title)
                # Manual adjustment instead of tight_layout
                fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
            else:
                # Sort data by count (descending) and take top 20
                sorted_data = sorted(data.items(), key=lambda x: x[1], reverse=True)[
                    :20
                ]

                if sorted_data:
                    classes = [item[0] for item in sorted_data]
                    counts = [item[1] for item in sorted_data]

                    # Create the bar plot
                    bars = ax.bar(
                        range(len(classes)), counts, color="steelblue", alpha=0.7
                    )

                    # Customize the plot
                    ax.set_xlabel(xlabel, fontsize=12)
                    ax.set_ylabel("Number of Images", fontsize=12)
                    ax.set_title(title, fontsize=14, pad=20)
                    ax.set_xticks(range(len(classes)))
                    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)

                    # Add value labels on bars
                    for bar, count in zip(bars, counts):
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height + max(counts) * 0.01,
                            f"{count}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )

                    # Manual adjustment with adequate space for rotated labels
                    fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.25)
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No data to plot",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(title)
                    fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)

            # Save to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", bbox_inches="tight", dpi=100)
            buffer.seek(0)

            # Convert to base64
            plot_data = base64.b64encode(buffer.getvalue()).decode()

            # Close the figure to free memory
            plt.close(fig)

            return plot_data

        except Exception as e:
            print(f"Error creating {plot_type} plot: {e}")
            # Create error plot
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(
                0.5,
                0.5,
                f"Error creating plot:\n{str(e)}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"Error in {plot_type} plot")
            fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)

            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", bbox_inches="tight")
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)

            return plot_data

    def create_confusion_matrix_plot(self, top_n=10):
        """Create confusion matrix for top misclassification pairs"""
        try:
            dist_data = self.get_distribution_data()
            confusion_pairs = dist_data["confusion_pairs"]

            # Clear any existing plots
            plt.clf()
            plt.close("all")

            # Get top N confusion pairs (already sorted by the serialization process)
            top_pairs = sorted(
                confusion_pairs.items(), key=lambda x: x[1], reverse=True
            )[:top_n]

            if not top_pairs:
                # Create empty plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(
                    0.5,
                    0.5,
                    "No confusion pairs available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title("Top Misclassification Patterns", fontsize=14)
                # Manual adjustment instead of tight_layout
                fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
            else:
                # Create figure with adequate space for labels
                fig, ax = plt.subplots(figsize=(16, max(8, len(top_pairs) * 0.6)))

                # Create labels and data for the plot
                pair_labels = [pair_str for pair_str, _ in top_pairs]
                counts = [count for _, count in top_pairs]

                # Create horizontal bar plot for better readability
                bars = ax.barh(
                    range(len(pair_labels)), counts, color="coral", alpha=0.7
                )

                # Customize the plot
                ax.set_xlabel("Number of Misclassifications", fontsize=12)
                ax.set_ylabel("True Class → Predicted Class", fontsize=12)
                ax.set_title("Top Misclassification Patterns", fontsize=14, pad=20)
                ax.set_yticks(range(len(pair_labels)))
                ax.set_yticklabels(pair_labels, fontsize=10)

                # Add value labels on bars
                for bar, count in zip(bars, counts):
                    width = bar.get_width()
                    ax.text(
                        width + max(counts) * 0.01,
                        bar.get_y() + bar.get_height() / 2.0,
                        f"{count}",
                        ha="left",
                        va="center",
                        fontsize=9,
                    )

                # Manual adjustment with adequate space for labels
                fig.subplots_adjust(left=0.35, right=0.95, top=0.92, bottom=0.08)

            # Save to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", bbox_inches="tight", dpi=100)
            buffer.seek(0)

            # Convert to base64
            plot_data = base64.b64encode(buffer.getvalue()).decode()

            # Close the figure to free memory
            plt.close(fig)

            return plot_data

        except Exception as e:
            print(f"Error creating confusion matrix plot: {e}")
            # Create error plot
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(
                0.5,
                0.5,
                f"Error creating plot:\n{str(e)}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Error in confusion matrix plot")
            fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)

            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", bbox_inches="tight")
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)

            return plot_data


# Global visualizer instance
visualizer = EditSetVisualizer()


@app.route("/")
def index():
    """Main dashboard page"""
    dist_data = visualizer.get_distribution_data()
    return render_template(
        "index.html",
        total_images=dist_data.get("total_images", 0),
        num_true_classes=len(dist_data.get("true_class_counts", {})),
        num_pred_classes=len(dist_data.get("pred_class_counts", {})),
    )


@app.route("/api/edit_sets")
def get_edit_sets():
    """API endpoint to get available edit sets"""
    return jsonify(visualizer.get_available_edit_sets())


@app.route("/api/edit_sets/<edit_set_name>", methods=["POST"])
def switch_edit_set(edit_set_name):
    """API endpoint to switch to a different edit set"""
    try:
        visualizer.load_edit_set(edit_set_name)
        return jsonify(
            {
                "success": True,
                "current_edit_set": edit_set_name,
                "message": f"Switched to {visualizer.available_edit_sets[edit_set_name]['display_name']}",
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/images")
def get_images():
    """API endpoint to get paginated images from edit set"""
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 12))

    if not visualizer.metadata or not visualizer.edit_data:
        return jsonify({"images": [], "total": 0, "pages": 0})

    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page

    images_data = []
    total_images = len(visualizer.metadata)

    for i in range(start_idx, min(end_idx, total_images)):
        meta = visualizer.metadata[i]

        # Get image tensor and convert to base64
        image_tensor = visualizer.edit_data["images"][i]
        image_b64 = visualizer.get_image_tensor_as_base64(image_tensor)

        # Get human-readable class names
        true_class_name = visualizer.get_class_name(meta["true_label"])
        predicted_class_name = visualizer.get_class_name(meta["predicted_label"])

        images_data.append(
            {
                "index": i,
                "image_data": image_b64,
                "true_class": meta["true_class"],
                "predicted_class": meta["predicted_class"],
                "true_label": true_class_name,
                "predicted_label": predicted_class_name,
                # Also include original labels if they exist in metadata
                "true_label_original": meta.get("true_label", true_class_name),
                "predicted_label_original": meta.get(
                    "predicted_label", predicted_class_name
                ),
            }
        )

    total_pages = (total_images + per_page - 1) // per_page

    return jsonify(
        {
            "images": images_data,
            "total": total_images,
            "pages": total_pages,
            "current_page": page,
        }
    )


@app.route("/api/distributions/true_classes")
def get_true_classes_plot():
    """API endpoint for true classes distribution plot"""
    plot_data = visualizer.create_distribution_plot("true_classes")
    return jsonify({"plot": plot_data})


@app.route("/api/distributions/predicted_classes")
def get_predicted_classes_plot():
    """API endpoint for predicted classes distribution plot"""
    plot_data = visualizer.create_distribution_plot("predicted_classes")
    return jsonify({"plot": plot_data})


@app.route("/api/distributions/confusion_matrix")
def get_confusion_matrix_plot():
    """API endpoint for confusion matrix plot"""
    plot_data = visualizer.create_confusion_matrix_plot()
    return jsonify({"plot": plot_data})


@app.route("/api/stats")
def get_stats():
    """API endpoint for summary statistics"""
    dist_data = visualizer.get_distribution_data()
    return jsonify(dist_data)


if __name__ == "__main__":
    # Create templates directory and HTML template
    os.makedirs("templates", exist_ok=True)

    # Run the Flask app
    print("Starting Edit Set Visualizer...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, host="0.0.0.0", port=5000)
