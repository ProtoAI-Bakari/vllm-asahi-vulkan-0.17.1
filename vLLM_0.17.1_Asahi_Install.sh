#!/bin/bash
set -euo pipefail

# Config
TAG="v0.17.1"
ENV_NAME=".venv-vLLM_0.17.1_Stable"
VENV_DIR="$HOME/$ENV_NAME"
REPO_DIR="$HOME/GITDEV/vllm_0.17.1"
TARGET_PYTHON="3.12"

echo "🚀 INITIATING vLLM $TAG NATIVE VULKAN DEPLOYMENT"

# 1. System Prep
sudo dnf install -y gcc-c++ cmake ninja-build rust cargo python3-devel \
                 vulkan-loader-devel mesa-vulkan-devel shaderc-devel libgomp
sudo ln -sf /usr/lib64/libgomp.so.1 /usr/lib64/libgomp.so

# 2. Workspace Prep
uv venv "$VENV_DIR" --python "$TARGET_PYTHON" --seed
source "$VENV_DIR/bin/activate"

# 3. Surgical Source Patches (The Connective Tissue)
cd "$REPO_DIR"

# Patch A: OpenMP discovery
sed -i 's/NO_DEFAULT_PATH//g' cmake/cpu_extension.cmake
sed -i 's/\${VLLM_TORCH_GOMP_SHIM_DIR}/\${VLLM_TORCH_GOMP_SHIM_DIR} \/usr\/lib64/g' cmake/cpu_extension.cmake

# Patch B: PyTorch 2.12 API alignment
python3 -c "
path = 'csrc/cpu/utils.hpp'
with open(path, 'r') as f: lines = f.readlines()
with open(path, 'w') as f:
    for l in lines:
        if 'at::cpu::L2_cache_size()' in l:
            f.write('#include <unistd.h>\n')
            f.write(l.replace('at::cpu::L2_cache_size()', '(sysconf(_SC_LEVEL2_CACHE_SIZE) > 0 ? sysconf(_SC_LEVEL2_CACHE_SIZE) : 4194304)'))
        else: f.write(l)
"

# 4. Hardware Build
export VLLM_VERSION="0.17.1+asahi"
export VLLM_TARGET_DEVICE="cpu"
export MAX_JOBS=$(nproc)
pip wheel . --no-build-isolation -w dist/

# 5. Final Lock-In
pip install --force-reinstall --no-deps ~/GITDEV/pytorch/dist/torch-2.12*.whl
pip install --force-reinstall dist/vllm-*.whl

echo "✅ HARDWARE STACK READY"
