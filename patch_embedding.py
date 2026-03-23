#!/usr/bin/env python3
"""
ASAHI VULKAN EMBEDDING FIX: Keep embedding weights on CPU permanently
to avoid Vulkan buffer allocation failure during weight transfer.
"""
import os

path = "/home/z/.venv-vLLM_0.17.1_Stable/lib/python3.12/site-packages/vllm/model_executor/layers/vocab_parallel_embedding.py"
with open(path, 'r') as f:
    content = f.read()

# Find the embedding method that does CPU↔Vulkan round-trip
old_code = """def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        if layer.weight.device.type == 'vulkan':
            # M1 MAX VULKAN WORKAROUND: Vulkan lacks index_select kernel
            # Hidden states lookup is performed on CPU and moved back to GPU
            return F.embedding(input_.to('cpu'), layer.weight.to('cpu')).to('vulkan')
        return F.embedding(input_, layer.weight)"""

new_code = """def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        if layer.weight.device.type == 'vulkan':
            # ASAHI VULKAN FIX: Vulkan buffer allocation fails for weight transfer
            # Keep embedding completely on CPU to avoid vmaCreateBuffer -2 error
            # Performance impact acceptable for small models on M1 Max
            return F.embedding(input_.to('cpu'), layer.weight.to('cpu')).to('cpu')
        return F.embedding(input_, layer.weight)"""

if old_code in content:
    content = content.replace(old_code, new_code)
    with open(path, 'w') as f:
        f.write(content)
    print("✅ vocab_parallel_embedding.py: Embedding layer pinned to CPU.")
    print("   Embedding output will remain on CPU (no Vulkan allocation).")
else:
    print("❌ ERROR: Could not find embedding method.")
    # Debug: show current content around line 75
    lines = content.split('\n')
    for i, line in enumerate(lines[70:85], start=70):
        print(f"Line {i}: {line}")