"""Quick CUDA / environment sanity check."""
import torch

print(f"torch            : {torch.__version__}")
print(f"compiled CUDA    : {torch.version.cuda}")
print(f"cuda available   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    i = torch.cuda.current_device()
    print(f"device name      : {torch.cuda.get_device_name(i)}")
    print(f"capability       : sm_{''.join(map(str, torch.cuda.get_device_capability(i)))}")
    props = torch.cuda.get_device_properties(i)
    print(f"total VRAM       : {props.total_memory / 1024**3:.1f} GB")
    # Prove a kernel actually runs on Blackwell:
    x = torch.randn(2048, 2048, device="cuda")
    y = (x @ x).sum().item()
    print(f"matmul on cuda   : OK ({y:.2e})")
else:
    print("!! CUDA not available — check torch cu128 install and NVIDIA driver.")
