"""
Script to install PyTorch with CUDA 12.x support
This will enable GPU training on RTX 3070 and RTX A5000
"""

import subprocess
import sys

print("=" * 80)
print("🔧 INSTALLING PYTORCH WITH CUDA 12.x SUPPORT")
print("=" * 80)
print()
print("Current issue: PyTorch is CPU-only")
print("Solution: Install PyTorch with CUDA 12.x support")
print()
print("This will:")
print("  1. Uninstall current CPU-only PyTorch")
print("  2. Install PyTorch with CUDA 12.x support")
print("  3. Enable GPU training (20-30x faster)")
print()
print("=" * 80)
print()

# Uninstall current PyTorch
print("Step 1: Uninstalling CPU-only PyTorch...")
subprocess.run([
    sys.executable, "-m", "pip", "uninstall", "-y",
    "torch", "torchvision", "torchaudio"
])

print()
print("Step 2: Installing PyTorch with CUDA 12.x support...")
print("(This may take a few minutes...)")
print()

# Install PyTorch with CUDA 12.x
# Using CUDA 12.1 as it's compatible with CUDA 12.9
subprocess.run([
    sys.executable, "-m", "pip", "install",
    "torch", "torchvision", "torchaudio",
    "--index-url", "https://download.pytorch.org/whl/cu121"
])

print()
print("=" * 80)
print("✅ INSTALLATION COMPLETE")
print("=" * 80)
print()
print("Verifying GPU detection...")
print()

# Verify installation
import torch
print(f"✅ CUDA available: {torch.cuda.is_available()}")
print(f"✅ GPU count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"✅ GPU {i}: {torch.cuda.get_device_name(i)}")
    print()
    print("🎉 SUCCESS! PyTorch can now use your GPUs!")
    print()
    print("Next steps:")
    print("  1. Stop the current CPU training (Ctrl+C in the training terminal)")
    print("  2. Run: python train_gpu.py")
    print("  3. Training will be 20-30x faster on RTX A5000!")
else:
    print()
    print("⚠️ WARNING: CUDA still not available")
    print("Please check:")
    print("  - NVIDIA drivers are installed")
    print("  - CUDA toolkit is installed")
    print("  - Restart your terminal/IDE")