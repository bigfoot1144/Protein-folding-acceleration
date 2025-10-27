import torch
torch._logging.set_logs(graph_breaks=True)
from torch.profiler import profile, record_function, ProfilerActivity
from inference import _load_model_and_alphabet_core_v2, MODEL_NAME
import os
import torch._dynamo as dynamo


def profile_inference():
    # Load model and data
    model_data = torch.load(os.path.join('model', f'{MODEL_NAME}.pt'),
                          mmap=True, weights_only=False)
    print("starting compile")
    model, alphabet, state_dict = _load_model_and_alphabet_core_v2(model_data)
    print("loading state dict")
    model.load_state_dict(state_dict, strict=False)
    
    # Create sample input
    sample_seq = "GMLSEQLKHCVVVGCGPAGLRTAIDLAKTLQIPEVISAARRTADKVKCVEATDLGFFEPQAASFLK"
    batch = [("sample", sample_seq)]
    _, _, tokens = alphabet.get_batch_converter()(batch)
    
    # Move model and inputs to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    tokens = tokens.to(device)
    model.eval()
    torch.set_grad_enabled(False)
    print("starting profile")
    rep = dynamo.explain(model, tokens)
    with open("graph_break_report.txt", "w") as f:
        f.write(str(rep))
    model = torch.compile(model, mode='reduce-overhead', dynamic=True)
    # Profile with PyTorch profiler
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA if torch.cuda.is_available() else None
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        with_flops=True
    ) as prof:
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                model(tokens)
        
        # Actual profiling
        with record_function("inference"):
            with torch.no_grad():
                out = model(tokens)
    

    # Print profiling results
    print("\n=== Profile Summary ===")
    print(prof.key_averages().table(
        sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
        row_limit=20
    ))
    
    # Export trace to Chrome trace viewer format
    prof.export_chrome_trace("inference_trace.json")
    
    # Memory summary
    print("\n=== Memory Summary ===")
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage" if torch.cuda.is_available() else "self_cpu_memory_usage"))

if __name__ == "__main__":
    profile_inference()


