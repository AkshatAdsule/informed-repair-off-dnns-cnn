# Specialized Repair Set Generation Utilities

This directory contains utilities for creating targeted repair sets that focus on specific types of misclassifications. These tools help create more focused datasets for neural network repair by categorizing errors based on the model's confidence and proximity to correct predictions.

## Overview

The utilities provide two main types of specialized repair sets:

1. **Confident Wrong** - Images where the model made incorrect predictions with high confidence
2. **Closest** - Misclassifications where the correct label had relatively high probability (almost correct)

## Scripts

### 1. `generate_confident_wrong_repairset.py`

Creates repair sets containing "confidently wrong" predictions where the classifier chose the wrong label with high confidence.

**Usage:**
```bash
python generate_confident_wrong_repairset.py [confidence_threshold] [max_samples]
```

**Parameters:**
- `confidence_threshold` (default: 0.8): Minimum confidence for wrong predictions
- `max_samples` (default: -1): Maximum number of samples to collect (-1 for unlimited)

**Example:**
```bash
# Generate confident wrong predictions with confidence >= 0.9
python generate_confident_wrong_repairset.py 0.9 1000

# Generate all confident wrong predictions with default threshold (0.8)
python generate_confident_wrong_repairset.py
```

**Output:**
- `data/edit_sets/confident_wrong_[threshold]_edit_dataset.pt`
- `data/edit_sets/confident_wrong_[threshold]_edit_metadata.json`

### 2. `generate_closest_repairset.py`

Creates repair sets containing misclassifications that were "almost correct" - where the true label had relatively high probability.

**Usage:**
```bash
python generate_closest_repairset.py [min_true_prob] [max_margin] [max_samples]
```

**Parameters:**
- `min_true_prob` (default: 0.1): Minimum probability for the true label
- `max_margin` (default: 0.3): Maximum margin between predicted and true probabilities
- `max_samples` (default: -1): Maximum number of samples to collect (-1 for unlimited)

**Example:**
```bash
# Generate closest misclassifications where true probability >= 0.15 and margin <= 0.2
python generate_closest_repairset.py 0.15 0.2 500

# Generate with default parameters
python generate_closest_repairset.py
```

**Output:**
- `data/edit_sets/closest_[min_prob]_[max_margin]_edit_dataset.pt`
- `data/edit_sets/closest_[min_prob]_[max_margin]_edit_metadata.json`

### 3. `generate_specialized_repairsets.py` (Recommended)

A comprehensive utility that combines both approaches and provides additional analysis capabilities.

**Usage:**
```bash
# Generate confident wrong repair set
python generate_specialized_repairsets.py confident_wrong [confidence_threshold] [max_samples]

# Generate closest repair set  
python generate_specialized_repairsets.py closest [min_true_prob] [max_margin] [max_samples]

# Generate both types (limited samples for demo)
python generate_specialized_repairsets.py both

# Analyze existing repair sets
python generate_specialized_repairsets.py analyze
```

**Examples:**
```bash
# Create confident wrong with 0.85 threshold, max 1000 samples
python generate_specialized_repairsets.py confident_wrong 0.85 1000

# Create closest with min_prob=0.2, max_margin=0.25, max 800 samples
python generate_specialized_repairsets.py closest 0.2 0.25 800

# Create both types with default parameters (500 samples each)
python generate_specialized_repairsets.py both

# Analyze all existing repair sets
python generate_specialized_repairsets.py analyze
```

## Key Concepts

### Confident Wrong Repair Sets

These repair sets focus on clear misclassifications where the model was very confident but wrong. This helps avoid ambiguous cases where multiple labels might be reasonable.

**Criteria:**
- Model prediction is incorrect
- Model confidence (max softmax probability) ≥ threshold

**Benefits:**
- Focuses on clear errors rather than ambiguous cases
- Helps identify systematic model failures
- Useful for targeted repair of confident mistakes

**Metadata includes:**
- `confidence`: Model's confidence in the wrong prediction
- `true_class` / `predicted_class`: Human-readable class names
- `type`: "confident_wrong"

### Closest Repair Sets

These repair sets contain misclassifications where the model was "close" to getting it right - the correct label had reasonably high probability.

**Criteria:**
- Model prediction is incorrect
- True label probability ≥ min_true_prob
- Margin (predicted_prob - true_prob) ≤ max_margin

**Benefits:**
- Focuses on cases where small adjustments might fix the error
- Identifies near-miss predictions
- Good candidates for fine-tuning approaches

**Metadata includes:**
- `true_probability`: Model's probability for the correct label
- `predicted_probability`: Model's probability for the predicted (wrong) label
- `margin`: Difference between predicted and true probabilities
- `type`: "closest"

## Data Format

All repair sets use the same format as the original edit sets:

**Dataset files (`.pt`):**
```python
{
    "images": torch.Tensor,      # Shape: [N, C, H, W] 
    "labels": torch.Tensor,      # Shape: [N] - true labels
    "metadata": List[Dict]       # Per-sample metadata
}
```

**Metadata structure:**
```json
{
    "image_idx": 1234,
    "true_label": 42,
    "predicted_label": 15,
    "true_class": "dalmatian",
    "predicted_class": "beagle",
    "type": "confident_wrong",
    
    // For confident_wrong:
    "confidence": 0.89,
    
    // For closest:
    "true_probability": 0.23,
    "predicted_probability": 0.41,
    "margin": 0.18
}
```

## Analysis and Statistics

Use the analyze command to get detailed statistics about existing repair sets:

```bash
python generate_specialized_repairsets.py analyze
```

This provides:
- Number of samples in each repair set
- Confidence/margin distributions
- Class diversity statistics
- File locations and sizes

## Integration with Existing Workflow

These specialized repair sets are compatible with the existing visualizer and experiment framework:

```python
# Load a specialized repair set
import torch
data = torch.load("data/edit_sets/confident_wrong_0.80_edit_dataset.pt")
images = data["images"]
labels = data["labels"] 
metadata = data["metadata"]

# Use with existing tools
python edit_set_visualizer.py  # Will show all repair sets including specialized ones
```

## Tips for Usage

1. **Confident Wrong Sets**: Start with threshold 0.8-0.9 to get clear mistakes
2. **Closest Sets**: Try min_prob=0.1-0.2 and max_margin=0.2-0.4 for good balance
3. **Sample Limits**: Use smaller limits (100-1000) for initial exploration
4. **Analysis**: Always run the analyze command to understand your data

## Examples of Good Parameters

**Conservative (high precision):**
```bash
python generate_specialized_repairsets.py confident_wrong 0.9 500
python generate_specialized_repairsets.py closest 0.2 0.2 500
```

**Balanced (good coverage):**
```bash
python generate_specialized_repairsets.py confident_wrong 0.8 1000  
python generate_specialized_repairsets.py closest 0.1 0.3 1000
```

**Exploratory (broader criteria):**
```bash
python generate_specialized_repairsets.py confident_wrong 0.7 2000
python generate_specialized_repairsets.py closest 0.05 0.4 2000
``` 