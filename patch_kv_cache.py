#!/usr/bin/env python3
import os

path = "/home/z/.venv-vLLM_0.17.1_Stable/lib/python3.12/site-packages/vllm/v1/core/kv_cache_utils.py"
with open(path, 'r') as f:
    content = f.read()

# The actual calculation pattern found in the file
old_code = """num_tokens = (
        kv_cache_config.num_blocks
        // len(kv_cache_config.kv_cache_groups)
        * min_block_size
    )"""

new_code = """# ASAHI VULKAN LOBOTOMY: Force tiny cache to prevent OOM
    if os.environ.get('VLLM_PLATFORM') == 'vulkan':
        num_tokens = 2048
    else:
        num_tokens = (
            kv_cache_config.num_blocks
            // len(kv_cache_config.kv_cache_groups)
            * min_block_size
        )"""

if old_code in content:
    content = content.replace(old_code, new_code)
    with open(path, 'w') as f:
        f.write(content)
    print("✅ kv_cache_utils.py: KV Cache tokens hard-capped at 2048 for Vulkan.")
else:
    print("❌ ERROR: Could not find exact calculation pattern.")
    # Show what we're looking for
    print("\nSearching for num_tokens assignment...")
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'num_tokens' in line and '=' in line:
            print(f"Line {i}: {line}")