# 🔴 VULKAN CRASH ANALYSIS: COMPREHENSIVE TECHNICAL REPORT
## For G3Pro Engineering Team

**Date:** March 22, 2026  
**Platform:** Asahi Linux (M1 Max)  
**vLLM Version:** 0.17.1  
**PyTorch Version:** Custom Build (Asahi Vulkan Backend)  
**Issue:** Vulkan dtype transfer crash during model initialization

---

## 1. EXECUTIVE SUMMARY

vLLM fails to load models on Asahi Linux with Vulkan backend due to **missing float16 support in PyTorch's Vulkan shader selection code**. The crash occurs during the `module.to('vulkan')` transfer in `device_loading_context()`, specifically in `Packing.cpp:65` where `get_nchw_to_image_shader()` rejects float16 tensors.

**Root Cause:** `get_nchw_to_image_shader()` only supports float32, bool, and quantized types. **float16 (kHalf) is NOT handled.**

**Impact:** All vLLM models using float16 weights (standard for most LLMs) cannot run on Vulkan backend.

---

## 2. FULL STACK TRACE

```
(EngineCore_DP0 pid=346745)   File "/home/z/.venv-vLLM_0.17.1_Stable/lib/python3.12/site-packages/vllm/model_executor/model_loader/utils.py", line 139, in device_loading_context
(EngineCore_DP0 pid=346745)     module.to('vulkan')
(EngineCore_DP0 pid=346745)   File "/home/z/.venv-vLLM_0.17.1_Stable/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1383, in to
(EngineCore_DP0 pid=346745)     return self._apply(convert)
(EngineCore_DP0 pid=346745)   File "/home/z/.venv-vLLM_0.17.1_Stable/lib/python3.12/site-packages/torch/nn/modules/module.py", line 964, in _apply
(EngineCore_DP0 pid=346745)     param_applied = fn(param)
(EngineCore_DP0 pid=346745)   File "/home/z/.venv-vLLM_0.17.1_Stable/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1369, in convert
(EngineCore_DP0 pid=346745)     return t.to(
(EngineCore_DP0 pid=346745) RuntimeError: Exception raised from get_nchw_to_image_shader at /home/z/GITDEV/pytorch/aten/src/ATen/native/vulkan/impl/Packing.cpp:65: Unsupported dtype!
```

---

## 3. CALL CHAIN CONNECTIVITY

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. vllm/v1/engine/core.py:834              EngineCoreProc.__init__()       │
│  2. vllm/v1/executor/abstract.py:103        _init_executor()                │
│  3. vllm/v1/executor/uniproc_executor.py:49 driver_worker.load_model()      │
│  4. vllm/v1/worker/gpu_worker.py:337        model_runner.load_model()       │
│  5. vllm/v1/worker/cpu_model_runner.py:63   self.model = get_model()        │
│  6. vllm/model_executor/model_loader/__init__.py:136  loader.load_model()   │
│  7. vllm/model_executor/model_loader/base_loader.py:74 process_weights()    │
│  8. vllm/model_executor/model_loader/utils.py:105 device_loading_context()  │
│  9. vllm/model_executor/model_loader/utils.py:139 module.to('vulkan')       │
│ 10. torch/nn/modules/module.py:1383         _apply(convert)                 │
│ 11. PyTorch C++ Backend                     to('vulkan')                    │
│ 12. pytorch/aten/src/ATen/native/vulkan/impl/Packing.cpp:65                 │
│     get_nchw_to_image_shader() → VK_THROW("Unsupported dtype!")             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. ROOT CAUSE ANALYSIS

### 4.1 The Failing Code (Packing.cpp:11-65)

```cpp
api::ShaderInfo get_nchw_to_image_shader(const vTensor& v_dst) {
  if (v_dst.is_quantized()) {
    // Quantized types supported (QUInt8, QInt8, QInt32)
    switch (v_dst.storage_type()) {
      case api::StorageType::TEXTURE_3D:
      case api::StorageType::TEXTURE_2D:
        // ... return shader for quantized types
    }
  }

  if (v_dst.dtype() == api::kFloat) {        // ✅ float32 SUPPORTED
    switch (v_dst.storage_type()) {
      case api::StorageType::TEXTURE_3D:
        return VK_KERNEL(nchw_to_image);
      case api::StorageType::TEXTURE_2D:
        return VK_KERNEL(nchw_to_image2d);
      default:
        VK_THROW("No kernel available!");
    }
  } else if (v_dst.dtype() == api::kBool) {  // ✅ bool SUPPORTED
    switch (v_dst.storage_type()) {
      case api::StorageType::TEXTURE_3D:
        return VK_KERNEL(nchw_to_image_bool);
      default:
        VK_THROW("No kernel available!");
    }
  } else {
    VK_THROW("Unsupported dtype!");          // ❌ float16 FAILS HERE
  }
}
```

### 4.2 Missing Case: float16 (api::kHalf)

The function has NO case for `api::kHalf` (float16). This is the root cause.

### 4.3 Why Float32 Bridge Works (Copy.cpp)

In `Copy.cpp`, float16 IS supported via a Float32 Bridge:

```cpp
// transfer_cpu_to_vulkan() in Copy.cpp
if (src.dtype() == at::kHalf) {
  // Convert to float32 before transfer
  memcpy_to_mapping(src_contig.to(at::kFloat), mapping);
} else {
  memcpy_to_mapping(src_contig, mapping);
}

// pack_vulkan_to_cpu() in Copy.cpp
if (dst.dtype() == at::kHalf) {
  Tensor dst_float = dst.to(at::kFloat);
  memcpy_from_mapping(mapping, dst_float);
  dst = dst_float.to(at::kHalf);  // Convert back to float16
}
```

**The Float32 Bridge works in Copy operations, but NOT in shader selection.**

---

## 5. WORKAROUNDS ATTEMPTED

| # | Workaround | Result | Status |
|---|------------|--------|--------|
| 1 | `PYTORCH_VULKAN_ENABLE_IMAGE_LAYOUT=0` | No effect | ❌ Failed |
| 2 | Shape-based env var toggle | No effect | ❌ Failed |
| 3 | Float32 Bridge (p.data.to(float32).to('vulkan').to(float16)) | Transfer succeeded, final cast failed | ⚠️ Partial |
| 4 | Restore clean `module.to('vulkan')` | Same crash | ❌ Failed |

**Why Workaround #3 Partially Worked:**
- `p.data.to(torch.float32).to('vulkan')` ✅ SUCCEEDS (float32 transfer works)
- `.to(torch.float16)` ❌ FAILS (Vulkan doesn't support float32→float16 conversion)

---

## 6. FILES INVOLVED

| # | File | Line(s) | Role |
|---|------|---------|------|
| 1 | `vllm/v1/engine/core.py` | 834 | EngineCoreProc initialization |
| 2 | `vllm/v1/executor/abstract.py` | 103 | Executor creation |
| 3 | `vllm/v1/executor/uniproc_executor.py` | 49 | Worker load trigger |
| 4 | `vllm/v1/worker/gpu_worker.py` | 337 | Model runner load |
| 5 | `vllm/v1/worker/cpu_model_runner.py` | 63 | get_model() call |
| 6 | `vllm/model_executor/model_loader/__init__.py` | 136 | Loader dispatch |
| 7 | `vllm/model_executor/model_loader/base_loader.py` | 74 | process_weights_after_loading |
| 8 | `vllm/model_executor/model_loader/utils.py` | 126-156 | device_loading_context |
| 9 | `vllm/model_executor/layers/vocab_parallel_embedding.py` | 112 | VocabParallelEmbedding class |
| 10 | `pytorch/aten/src/ATen/native/vulkan/impl/Packing.cpp` | 11-65 | **ROOT CAUSE** - Shader selection |
| 11 | `pytorch/aten/src/ATen/native/vulkan/ops/Copy.cpp` | 168-223 | Float32 Bridge (works for copy, not transfer) |
| 12 | `pytorch/aten/src/ATen/native/vulkan/ops/Utils.cpp` | 205 | Calls get_nchw_to_image_shader() |

---

## 7. TECHNICAL DIAGRAM

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  FLOAT16 TRANSFER FLOW                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CPU Side                    Vulkan Backend                                  │
│  ─────────                   ───────────                                     │
│  float16 tensor               ┌─────────────────────────────┐               │
│       │                       │ module.to('vulkan')         │               │
│       │                       └──────────┬──────────────────┘               │
│       │                                  │                                   │
│       │                                  ▼                                   │
│       │                       ┌─────────────────────────────┐               │
│       │                       │ pack_cpu_to_vulkan()        │               │
│       │                       └──────────┬──────────────────┘               │
│       │                                  │                                   │
│       │                                  ▼                                   │
│       │                       ┌─────────────────────────────┐               │
│       │                       │ pack_buffer_to_vtensor()    │               │
│       │                       └──────────┬──────────────────┘               │
│       │                                  │                                   │
│       │                                  ▼                                   │
│       │                       ┌─────────────────────────────┐               │
│       │                       │ get_nchw_to_image_shader()  │               │
│       │                       │   if dtype == kFloat ✅     │               │
│       │                       │   if dtype == kBool ✅      │               │
│       │                       │   else ❌ "Unsupported!"    │               │
│       │                       └──────────┬──────────────────┘               │
│       │                                  │                                   │
│       │                                  ▼                                   │
│       │                       ┌─────────────────────────────┐               │
│       │                       │ VK_THROW("Unsupported dtype!")              │
│       │                       └─────────────────────────────┘               │
│       │                                                                     │
│  SOLUTION: Add kHalf case to get_nchw_to_image_shader()                     │
│  OR: Force BUFFER storage instead of TEXTURE                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. RECOMMENDED FIXES

### Option A: Add float16 Support to Packing.cpp (RECOMMENDED)

Add a `kHalf` case to `get_nchw_to_image_shader()`:

```cpp
} else if (v_dst.dtype() == api::kHalf) {  // ADD THIS CASE
  switch (v_dst.storage_type()) {
    case api::StorageType::TEXTURE_3D:
      return VK_KERNEL(nchw_to_image_half);  // Need to implement this shader
    case api::StorageType::TEXTURE_2D:
      return VK_KERNEL(nchw_to_image2d_half);  // Need to implement this shader
    default:
      VK_THROW("No kernel available!");
  }
}
```

**Pros:** Clean, proper support  
**Cons:** Requires implementing new Vulkan shaders

### Option B: Force BUFFER Storage for float16

Modify the tensor creation to use BUFFER storage type for float16 tensors:

```cpp
// In tensor creation code
if (dtype == api::kHalf) {
  storage_type = api::StorageType::BUFFER;  // Force buffer instead of texture
}
```

**Pros:** Uses existing Copy.cpp float16 support  
**Cons:** May impact performance for some operations

### Option C: Pre-convert Model to float32

Force vLLM to load models in float32:

```bash
vllm serve model --dtype float32
```

**Pros:** No code changes  
**Cons:** 2x memory usage, slower inference

### Option D: Use CPU Backend

Disable Vulkan backend entirely:

```bash
export VLLM_USE_VULKAN=0
```

**Pros:** Works immediately  
**Cons:** No Vulkan acceleration

---

## 9. TESTING VERIFICATION

### Current State
```bash
# Test Vulkan availability
python3 -c "import torch; print('Vulkan:', torch.vulkan.is_available())"

# Check dtype support
python3 -c "import torch; t = torch.randn(10, 10, dtype=torch.float16); print(t.to('vulkan'))"
# Expected: RuntimeError: Unsupported dtype!
```

### After Fix
```bash
# Should succeed
python3 -c "import torch; t = torch.randn(10, 10, dtype=torch.float16); print(t.to('vulkan'))"
# Expected: tensor on vulkan device with float16 dtype
```

---

## 10. CONCLUSION

**Root Cause:** PyTorch Vulkan backend's `get_nchw_to_image_shader()` function lacks float16 (kHalf) support.

**Impact:** All vLLM models using float16 weights fail to load on Asahi Linux Vulkan backend.

**Recommended Fix:** Add kHalf case to `get_nchw_to_image_shader()` in `Packing.cpp` OR force BUFFER storage for float16 tensors.

**Timeline:** Requires PyTorch Vulkan backend modification. Not a vLLM issue.

---

## 11. CONTACT INFORMATION

**Investigation Lead:** Z-Alpha Autonomous Agent  
**Platform:** Asahi Linux (M1 Max)  
**Date:** March 22, 2026  
**Status:** Root Cause Identified, Fix Options Documented

---

*End of Technical Summary*
