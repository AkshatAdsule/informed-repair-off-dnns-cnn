import torch
import sytorch as st
import os
from helpers.dataset import imagenet_mini
from helpers.eval import evaluate
from helpers.models import squeezenet

# SyTorch setup
device = st.device("cpu")
dtype = st.float64


def repair_squeezenet_on_edit_image():
    # Load and convert model
    print("Loading and converting SqueezeNet model...")
    torch_model = squeezenet(pretrained=True, eval=True)
    model = torch_model.to(dtype=dtype, device=device)
    model.eval()

    # Load edit set
    print("Loading edit set...")
    edit_data = torch.load(
        "data/edit_sets/squeezenet_edit_dataset.pt", map_location=device
    )
    edit_images = edit_data["images"].to(dtype=dtype, device=device)
    edit_labels = edit_data["labels"].to(device=device)

    # Get first image and label
    image_to_repair = edit_images[0].unsqueeze(0)
    true_label = edit_labels[0].item()

    # Print image details if metadata exists
    if edit_data.get("metadata") and len(edit_data["metadata"]) > 0:
        meta = edit_data["metadata"][0]
        print(
            f"Image: True Class '{meta.get('true_class')}' (Label: {meta.get('true_label')})"
        )

    # Evaluate original model
    print("\nEvaluating original model...")
    with st.no_grad():
        original_output = model(image_to_repair)
        _, original_pred = torch.max(original_output, 1)
        print(f"Original prediction: {original_pred.item()} (True: {true_label})")

    # Setup repair
    print("\nSetting up repair...")
    solver = st.GurobiSolver()
    N = model.deepcopy().to(solver).repair()

    # Make classifier layer symbolic
    param_change_bound = 15.0
    layer_to_repair = N[1][1]  # Classifier Conv2d layer
    layer_to_repair.weight.requires_symbolic_(
        lb=-param_change_bound, ub=param_change_bound
    )
    if layer_to_repair.bias is not None:
        layer_to_repair.bias.requires_symbolic_(
            lb=-param_change_bound, ub=param_change_bound
        )

    # Define constraints
    print("Defining constraints...")
    with st.no_symbolic(), st.no_grad():
        reference_output = N(image_to_repair)

    symbolic_output = N(image_to_repair)
    true_class_logit = symbolic_output[0, true_label]
    margin = 15.0

    constraints = []
    for i in range(1000):  # ImageNet classes
        if i != true_label:
            constraints.append(true_class_logit >= symbolic_output[0, i] + margin)

    # Solve
    print("\nSolving repair problem...")
    param_deltas = N.parameter_deltas(concat=True)
    output_deltas = (symbolic_output - reference_output).flatten().alias()
    objective = st.cat([output_deltas, param_deltas]).norm_ub("linf+l1_normalized")

    if solver.solve(*constraints, minimize=objective):
        print("Repair successful!")
        N.update_()
        N.repair(False)
        N.eval()

        # Evaluate repaired model
        with st.no_grad():
            repaired_output = N(image_to_repair)
            _, repaired_pred = torch.max(repaired_output, 1)
            print(f"Repaired prediction: {repaired_pred.item()} (True: {true_label})")
    else:
        print("Repair failed - could not find solution.")


if __name__ == "__main__":
    repair_squeezenet_on_edit_image()
