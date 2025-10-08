import torch
import importlib
import argparse

def test_numerical_equivalency(model_name: str):
    """
    Tests numerical equivalency between a model and its 'fast' version.

    Args:
        model_name (str): The name of the model module to test (e.g., 'esm2').
    """
    try:
        # Dynamically import the original and fast model modules
        original_module = importlib.import_module(f"{model_name.split('.')[0]}.{model_name.split('.')[1]}")
        fast_module = importlib.import_module(f"{model_name.split('.')[0]}_fast.{model_name.split('.')[1]}")
    except ImportError as e:
        print(f"Could not import modules for {model_name}. Error: {e}")
        return

    # Get the class from the module (assuming class name is the same as file name, capitalized)
    # e.g. esm2.py -> ESM2
    ClassName = model_name.split('.')[1].upper()
    OriginalModel = getattr(original_module, ClassName)
    FastModel = getattr(fast_module, ClassName)

    # Instantiate the models
    # Its default values should work on their own
    original_model = OriginalModel()
    fast_model = FastModel()

    original_model.eval()
    fast_model.eval()

    # Get a random input tensor.
    # This assumes you've implemented a `get_random_input` method on your nn.Module.
    try:
        # We can get the input from either model, assuming they expect the same input.
        input_tensor = original_model.get_random_input()
    except AttributeError:
        print(f"Please implement `get_random_input(self)` method in your {ClassName} model.")
        print("It should return a random tensor of the expected input shape for the model.")
        return

    # Forward pass
    with torch.no_grad():
        original_output = original_model(input_tensor)
        fast_output = fast_model(input_tensor)

    # Check shape equivalence
    if original_output.shape != fast_output.shape:
        print(f"❌ Output shapes differ: {original_output.shape} vs {fast_output.shape}")
        return

    # Check dtype equivalence
    if original_output.dtype != fast_output.dtype:
        print(f"❌ Output dtypes differ: {original_output.dtype} vs {fast_output.dtype}")
        return

    # Check for numerical equivalency
    # torch.allclose is used to compare floating point tensors for near-equality.
    are_equivalent = torch.allclose(original_output, fast_output)

    if are_equivalent:
        print(f"✅ Outputs of {ClassName} and {ClassName}_fast are numerically equivalent.")
    else:
        print(f"❌ Outputs of {ClassName} and {ClassName}_fast are NOT numerically equivalent.")
        diff = torch.abs(original_output - fast_output).max()
        print(f"   Max absolute difference: {diff.item()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run numerical equivalency tests for models.")
    parser.add_argument(
        "model",
        type=str,
        help="The model to test, e.g., 'esm.esm2'"
    )
    args = parser.parse_args()

    test_numerical_equivalency(args.model)
