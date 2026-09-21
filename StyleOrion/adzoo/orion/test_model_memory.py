# ==============================================================================
# [Bench2Drive-Style] NEW FILE -- script
# Author: YinXuan1223   (carried over from the ORION_NYCUACM_version fork)
#
# Throwaway diagnostic used while bringing the model up on 24 GB GPUs: builds the
# model on CPU, prints the parameter/buffer counts and the per-dtype memory split,
# loads a checkpoint and converts to bfloat16 to confirm the footprint halves.
# The repo root is derived from the file's own location; the checkpoint path is still
# hard-coded. A scratch tool, not part of any pipeline.
# ==============================================================================
"""Quick test: measure model memory footprint before inference."""
import torch
import sys
import os
# [Bench2Drive-Style] repo root derived from this file's location
# (adzoo/orion/ -> ../..), so it no longer hard-codes the old 'Ourion' path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from mmcv.utils import Config, load_checkpoint
from mmcv.models import build_model

cfg = Config.fromfile('adzoo/orion/configs/orion_stage3_fp16_diff.py')
cfg.model.pretrained = None
cfg.model.train_cfg = None

print("Building model on CPU...")
model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
total_buffers = sum(b.numel() for b in model.buffers())
print(f"Total parameters: {total_params/1e9:.2f}B")
print(f"Total buffers: {total_buffers/1e6:.2f}M")

# Check dtypes
dtypes = {}
for name, p in model.named_parameters():
    dtype = str(p.dtype)
    if dtype not in dtypes:
        dtypes[dtype] = 0
    dtypes[dtype] += p.numel()
print("\nParameter dtypes:")
for dtype, count in dtypes.items():
    size_gb = count * (2 if '16' in dtype else 4) / 1e9
    print(f"  {dtype}: {count/1e9:.2f}B params = {size_gb:.2f} GB")

print("\nLoading checkpoint...")
ckpt = load_checkpoint(model, 'adzoo/orion/work_dirs/orion_stage2_diff_train/iter_30000.pth', map_location='cpu')

print("\nConverting to bfloat16...")
model = model.to(torch.bfloat16)

# Re-check dtypes after conversion
dtypes2 = {}
for name, p in model.named_parameters():
    dtype = str(p.dtype)
    if dtype not in dtypes2:
        dtypes2[dtype] = 0
    dtypes2[dtype] += p.numel()
print("Parameter dtypes after bf16 conversion:")
for dtype, count in dtypes2.items():
    size_gb = count * (2 if '16' in dtype else 4) / 1e9
    print(f"  {dtype}: {count/1e9:.2f}B params = {size_gb:.2f} GB")

total_bf16_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
total_buf_gb = sum(b.numel() * b.element_size() for b in model.buffers()) / 1e9
print(f"\nTotal model memory (params): {total_bf16_gb:.2f} GB")
print(f"Total model memory (buffers): {total_buf_gb:.4f} GB")
print(f"Total: {total_bf16_gb + total_buf_gb:.2f} GB")

print("\nMoving to GPU...")
torch.cuda.reset_peak_memory_stats()
model = model.cuda()
print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"GPU memory reserved:  {torch.cuda.memory_reserved()/1e9:.2f} GB")
print(f"GPU memory free:      {(24576*1024*1024 - torch.cuda.memory_allocated())/1e9:.2f} GB")

model.eval()
print("\nModel loaded successfully. Ready for inference test.")
del model
torch.cuda.empty_cache()
