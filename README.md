# Informed Repair of DNNs

## Overview

The experimental setup evaluates **5 main categories of heuristics**:

1. **Activation-Based Heuristics** - Select layers based on activation statistics
2. **Gradient/Sensitivity-Based Heuristics** - Identify layers with highest parameter sensitivity  
3. **Feature-Similarity Based Heuristics** - Choose layers with similar internal representations
4. **Layer Type/Position Heuristics** - Simple structural selection strategies
5. **Neuron/Path Contribution Heuristics** - Select based on neuron firing patterns and path utilization

## Implemented Heuristics

### 1. Activation-Based Heuristics

*"Selecting the start layer based on metrics calculated across the repair set, such as the layer exhibiting the highest average activation magnitude, or perhaps the layer with the highest variance in activations."*

- `ActivationMagnitudeHeuristic` - Selects layers with highest activation magnitudes
- `ActivationVarianceHeuristic` - Selects layers with highest activation variance

### 2. Gradient/Sensitivity-Based Heuristics

*"Identifying layers where parameters show the most sensitivity (e.g., largest gradient norms) with respect to the inputs in the repair set."*

- `GradientNormHeuristic` - Layers with highest gradient norms
- `ParameterSensitivityHeuristic` - Layers with highest parameter sensitivity
- `OutputSensitivityHeuristic` - Layers most directly influencing output

### 3. Feature-Similarity Based Heuristics

*"Choosing a layer where the internal representations of the inputs within the repair set are deemed most similar, suggesting a point of unified processing relevant to the required fix."*

- `FeatureConvergenceHeuristic` - Layers where features converge
- `FeatureDivergenceHeuristic` - Layers where features diverge
- `BottleneckHeuristic` - Squeeze layers with natural feature similarity

### 4. Layer Type/Position Heuristics

*"Simple heuristics like always choosing the first fully-connected layer after convolutional blocks, or the penultimate layer."*

- `PenultimateLayerHeuristic` - Last layers before classifier
- `FirstFCHeuristic` - First fully-connected-like layer  
- `TransitionLayerHeuristic` - Layers around pooling/transition points

### 5. Neuron/Path Contribution Heuristics

*"Run inference over the input polytope defined by the repair set and track neuron activations across the network. Aggregate activation patterns to identify neurons or layers that consistently fire across the polytope."*

- `MagnitudeContributionHeuristic` - Highest activation contribution
- `VarianceContributionHeuristic` - Highest variance in contributions
- `ConsistencyContributionHeuristic` - Most consistent activation patterns
- `PathStrengthHeuristic` - Strongest activation paths

## Edit Set Visualizer

The project includes a web-based visualizer for exploring different edit sets containing misclassified images. 

### Features

- **Multi-Edit Set Support**: Switch between different edit sets via dropdown
- **Image Browser**: Paginated gallery of misclassified images with true/predicted labels
- **Distribution Analysis**: Charts showing class distributions and confusion patterns
- **Summary Statistics**: Overview of misclassification patterns and dataset stats

### Available Edit Sets

1. **Squeezenet Edit** (1,713 images, 983.7MB): Full edit set from SqueezeNet
2. **Confident Wrong 0.85 Edit** (70 images, 40.2MB): High-confidence misclassifications
3. **Closest 0.15 0.20 Edit** (202 images, 116.0MB): Specialized repair set with confidence margins

### Usage

```bash
# Start the visualizer
uv run python edit_set_visualizer.py

# Open browser to http://localhost:5000
```

### Edit Set Switching

Use the dropdown at the top of the page to switch between different edit sets. The interface will automatically reload to show:

- Updated image counts and statistics
- New distribution plots for the selected edit set
- Fresh image gallery with pagination
- Recalculated confusion matrices and analysis