# Vulkan Buffer Allocation Fix Analysis

## Problem
vmaCreateBuffer returns -2 (VMA_ERROR_OUT_OF_DEVICE_MEMORY) during embedding layer warmup

## Root Cause Chain
1. Vulkan backend lacks index_select kernel on Asahi Linux
2. vLLM fallback moves embedding to CPU then back to Vulkan
3. Buffer allocation fails due to memory constraints/fragmentation

## Solutions

### Option 1: Increase Memory Utilization
```bash
export gpu_memory_utilization=0.8  # Was 0.3
```

### Option 2: Reduce Model Length
```bash
export max_model_len=256  # Was 512
```

### Option 3: Disable CPU Fallback (if possible)
Patch vocab_parallel_embedding.py to not force CPU round-trip

### Option 4: Use CPU Worker Instead
```bash
export VLLM_PLATFORM=cpu
```

## Immediate Action Required
1. Increase gpu-memory-utilization to 0.6-0.8
2. Verify Vulkan memory is not fragmented from previous runs
3. Consider reducing max-model-len to 256
