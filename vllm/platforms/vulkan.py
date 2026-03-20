import os, sys, torch, types, gc, psutil, contextlib
from vllm.platforms.interface import Platform, PlatformEnum

# 1. STARVE KV CACHE ALLOCATOR
import vllm.utils
vllm.utils.get_cpu_memory = lambda: 512 * 1024**2  

_orig_vmem = psutil.virtual_memory
def _fake_vmem():
    mem = _orig_vmem()
    class VMem:
        total = mem.total
        available = 512 * 1024**2
        percent = 99.0
        used = mem.total - (512 * 1024**2)
        free = 512 * 1024**2
    return VMem()
psutil.virtual_memory = _fake_vmem

# 2. BF16 KILLER & INSTANT STORAGE NUKE
_orig_to = torch.Tensor.to
VULKAN_SUPPORTED = {torch.float32, torch.float16}

def _vto(self, *args, **kwargs):
    device = kwargs.get("device") or (args[0] if args else None)
    if device is not None and "vulkan" in str(device).lower():
        target = self
        if target.dtype == torch.bfloat16:
            target = _orig_to(target, dtype=torch.float16, device="cpu")
        try:
            gpu_tensor = _orig_to(target, "vulkan")
            if self.device.type == "cpu" and self.numel() > 0:
                try: self.untyped_storage().resize_(0)
                except Exception: pass
            gc.collect()
            return gpu_tensor
        except Exception: return self
    return _orig_to(self, *args, **kwargs)
torch.Tensor.to = _vto

# 3. CUDA FAKES (Bulletproofed for V1)
if not hasattr(torch, 'cuda'): torch.cuda = types.ModuleType("torch.cuda")
torch.cuda.is_available = lambda: True
torch.cuda.get_device_properties = lambda d: type("P", (), {"total_memory": 16*1024**3, "major": 8, "minor": 0})()
torch.cuda.mem_get_info = lambda d=0: (512 * 1024**2, 16 * 1024**3)
torch.cuda.current_device = lambda: 0
torch.cuda.set_device = lambda d: None

class DStream:
    def __init__(self, *args, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def wait_stream(self, *a, **k): pass
    def __getattr__(self, n): return lambda *a, **k: None

torch.cuda.Stream = DStream
torch.cuda.current_stream = lambda *a, **k: DStream()
torch.cuda.default_stream = lambda *a, **k: DStream()
torch.cuda.stream = lambda *a, **k: DStream()
torch.cuda.Event = DStream

# 4. PIN_MEMORY STRIPPER
_ot = torch.tensor
torch.tensor = lambda *a, **k: _ot(*a, **{x:y for x,y in k.items() if x != 'pin_memory'})
def anchor(f):
    def w(*a, **k):
        k.pop("pin_memory", None)
        dev = str(k.get("device", ""))
        if "vulkan" not in dev.lower(): k["device"] = "cpu"
        return f(*a, **k)
    return w
for op in ['arange', 'zeros', 'ones', 'empty', 'full']:
    setattr(torch, op, anchor(getattr(torch, op)))

# 5. V1 PLATFORM DEFINITION
class VulkanPlatform(Platform):
    _enum = PlatformEnum.VULKAN
    device_name, device_type, dispatch_key = "vulkan", "cpu", "Vulkan"
    
    @classmethod
    def get_device_name(cls, d=0): return "cpu"
    @classmethod
    def get_default_worker_cls_name(cls): return "vllm.v1.worker.cpu_worker.CPUWorker"
    @classmethod
    def get_default_model_runner_cls_name(cls): return "vllm.v1.worker.cpu_model_runner.CPUModelRunner"
    
    @classmethod
    def get_attn_backend_cls(cls, *a, **k):
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
        return AttentionBackendEnum.CPU_ATTN.get_path()

    def is_vulkan(self): return True
    @classmethod
    def num_compute_units(cls, d=0): return 32