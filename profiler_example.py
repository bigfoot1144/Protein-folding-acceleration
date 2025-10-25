# profiler_step2.py
import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity, record_function

# ---- pick device + activities ----
activities = [ProfilerActivity.CPU]
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
    activities.append(ProfilerActivity.CUDA)
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    device = "xpu"
    activities.append(ProfilerActivity.XPU)

# ---- model + input ----
model = models.resnet18().to(device).eval()
inputs = torch.randn(5, 3, 224, 224, device=device)

# 1) TIME PROFILE (same as before)
with profile(activities=activities, record_shapes=True) as prof:
    with record_function("model_inference"):
        with torch.inference_mode():
            model(inputs)

sort_key = ("cpu_time_total" if device == "cpu" else device + "_time_total")
print("\n== Time profile ==")
print(prof.key_averages().table(sort_by=sort_key, row_limit=10))

# 2) MEMORY PROFILE (CPU tensor allocations)
with profile(
    activities=[ProfilerActivity.CPU],
    profile_memory=True,
    record_shapes=True,
) as mem_prof:
    with torch.inference_mode():
        models.resnet18()(torch.randn(5, 3, 224, 224))  # CPU-only to keep it simple

print("\n== Memory profile (by self_cpu_memory_usage) ==")
print(mem_prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

# 3) CHROME TRACE EXPORT (for timeline view)
with profile(activities=activities) as trace_prof:
    with torch.inference_mode():
        model(inputs)

trace_prof.export_chrome_trace("trace.json")
print("\nWrote Chrome trace to trace.json (open with chrome://tracing)")
