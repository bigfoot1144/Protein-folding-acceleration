import torch
import importlib
import argparse
import numpy as np
from typing import List

def snake_to_camel(snake_case_name: str) -> str:
    """Converts a snake_case string to CamelCase."""
    # Special case for 'esm2' -> 'ESM2'
    if snake_case_name == 'esm2':
        return 'ESM2'
    return "".join(word.capitalize() for word in snake_case_name.split('_'))

def benchmark(model_name: str, iterations: int, warmup: int):
    """
    Benchmarks a model against its 'fast' version for runtime and memory.

    Args:
        model_name (str): The name of the model module (e.g., 'esm.esm2').
        iterations (int): The number of benchmark iterations.
        warmup (int): The number of warmup iterations.
    """
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. This benchmark requires a GPU.")
        return

    print(f"Benchmarking {model_name} vs. {model_name.replace('.', '_fast.')}")

    try:
        module_path, model_file = model_name.split('.')
        original_module = importlib.import_module(f"{module_path}.{model_file}")
        fast_module = importlib.import_module(f"{module_path}_fast.{model_file}")
    except ImportError as e:
        print(f"Could not import modules for {model_name}. Error: {e}")
        return

    ClassName = snake_to_camel(model_file)
    try:
        OriginalModel = getattr(original_module, ClassName)
        FastModel = getattr(fast_module, ClassName)
    except AttributeError:
        print(f"❌ Could not find class '{ClassName}' in the imported modules.")
        print("   Please ensure your class name follows CamelCase convention (e.g., 'Esm2', 'MultiheadAttention').")
        return

    # Instantiate and prepare models
    original_model = OriginalModel().to('cuda').eval()
    fast_model = FastModel().to('cuda').eval()

    try:
        input_tensor = original_model.get_random_input().to('cuda')
    except AttributeError:
        print(f"❌ Please implement `get_random_input(self)` in your {ClassName} model.")
        return

    # --- Warmup Phase ---
    print(f"Warming up for {warmup} iterations...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = original_model(input_tensor)
            _ = fast_model(input_tensor)
    torch.cuda.synchronize()

    # --- Benchmark Phase ---
    original_runtimes: List[float] = []
    fast_runtimes: List[float] = []
    original_memories: List[float] = []
    fast_memories: List[float] = []

    print(f"Running benchmark for {iterations} iterations...")
    for i in range(iterations):
        # --- Original Model Run ---
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        with torch.no_grad():
            start_event.record()
            _ = original_model(input_tensor)
            end_event.record()
        
        torch.cuda.synchronize()
        original_runtimes.append(start_event.elapsed_time(end_event))
        original_memories.append(torch.cuda.max_memory_allocated() / 1e6) # Convert to MB

        # --- Fast Model Run ---
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            start_event.record()
            _ = fast_model(input_tensor)
            end_event.record()

        torch.cuda.synchronize()
        fast_runtimes.append(start_event.elapsed_time(end_event))
        fast_memories.append(torch.cuda.max_memory_allocated() / 1e6) # Convert to MB
        
        print(f"  Iteration {i + 1}/{iterations} complete.", end='\r')
    print("\n")


    # --- Report Results ---
    def print_stats(model_title: str, runtimes: List[float], memories: List[float]):
        print(f"\n--- {model_title} ---")
        print(f"  Avg Runtime: {np.mean(runtimes):.3f} ms (std: {np.std(runtimes):.3f} ms)")
        print(f"  Avg Max Memory: {np.mean(memories):.2f} MB (std: {np.std(memories):.2f} MB)")

    print_stats(f"Original Model ({ClassName})", original_runtimes, original_memories)
    print_stats(f"Fast Model ({ClassName})", fast_runtimes, fast_memories)

    runtime_improvement = (np.mean(original_runtimes) - np.mean(fast_runtimes)) / np.mean(original_runtimes) * 100
    print(f"\n--- Summary ---")
    print(f"🚀 Runtime Improvement: {runtime_improvement:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark a model against its 'fast' version.")
    parser.add_argument("model", type=str, help="The model to benchmark, e.g., 'esm.esm2'")
    parser.add_argument("-i", "--iterations", type=int, default=10, help="Number of benchmark iterations")
    parser.add_argument("-w", "--warmup", type=int, default=3, help="Number of warmup iterations")
    args = parser.parse_args()

    benchmark(args.model, args.iterations, args.warmup)
