import os, sys, torch, types, gc, psutil
from vllm.platforms.interface import Platform, PlatformEnum

# --- 1. System Memory Lie ---
import vllm.utils
vllm.utils.get_cpu_memory = lambda: 512 * 1024**2
_orig_vmem = psutil.virtual_memory
psutil.virtual_memory = lambda: type("VMem", (), {
    "total": _orig_vmem().total, "available": 512*1024**2, "percent": 99.0,
    "used": _orig_vmem().total - (512*1024**2), "free": 512*1024**2
})()

# --- 2. Vulkan Tensor Offloading Interceptor ---
_orig_to = torch.Tensor.to
def _vto(self, *args, **kwargs):
    device = kwargs.get("device") or (args[0] if args else None)
    if device is not None and "vulkan" in str(device).lower():
        target = self.to(dtype=torch.float16, device="cpu") if self.dtype == torch.bfloat16 else self
        try:
            res = _orig_to(target, "vulkan")
            if self.device.type == "cpu" and self.numel() > 0:
                try: self.untyped_storage().resize_(0)
                except: pass
            return res
        except: return self
    return _orig_to(self, *args, **kwargs)
torch.Tensor.to = _vto

# --- 3. CUDA Fake API ---
if not hasattr(torch, 'cuda'): torch.cuda = types.ModuleType("torch.cuda")
torch.cuda.is_available = lambda: True
torch.cuda.get_device_properties = lambda d: type("P", (), {"total_memory": 16*1024**3, "major": 8, "minor": 0})()
torch.cuda.mem_get_info = lambda d=0: (512*1024**2, 16*1024**3)
class DStream:
    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def wait_stream(self, *a, **k): pass
    def __getattr__(self, n): return lambda *a, **k: None
torch.cuda.Stream = torch.cuda.Event = DStream
torch.cuda.current_stream = torch.cuda.default_stream = lambda *a, **k: DStream()

# --- 4. Platform Definition (Stealth CPU Mode) ---
class VulkanPlatform(Platform):
    def __init__(self):
        self.device_name = "cpu"
        self.device_type = "cpu"
        self.dispatch_key = "Vulkan"

    _enum = PlatformEnum.VULKAN
    device_name: str = "vulkan"
    device_type: str = "cpu"  # THE BYPASS: Force vLLM to treat us as CPU for config
    dispatch_key: str = "Vulkan"

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str: return "cpu"
    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool) -> bool: return False
    @classmethod
    def get_default_worker_cls_name(cls) -> str: return "vllm.v1.worker.cpu_worker.CPUWorker"
    @classmethod
    def get_default_model_runner_cls_name(cls) -> str: return "vllm.v1.worker.cpu_model_runner.CPUModelRunner"
    @classmethod
    def get_attn_backend_cls(cls, *a, **k) -> str:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
        return AttentionBackendEnum.CPU_ATTN.get_path()
    
    # Identify as CPU to pass all v0.17.1 config guards
    def is_cpu(self) -> bool: return True
    def is_vulkan(self) -> bool: return True
    def is_cuda(self) -> bool: return False
    def is_rocm(self) -> bool: return False
    
    @classmethod
    def get_cpu_architecture(cls):
        from vllm.platforms.interface import CpuArchEnum
        return CpuArchEnum.ARM

    # --- 5. v0.17.1 Config Survival Stubs ---
    simple_compile_backend = "eager"
    
    @classmethod
    def get_compile_backend(cls): return "eager"
    @classmethod
    def get_pass_manager_cls(cls): return "vllm.compilation.passes.pass_manager.PostGradPassManager"
    @property
    def pass_key(self): return "post_grad_custom_post_pass"
    @classmethod
    def is_pin_memory_available(cls): return False
    @classmethod
    def get_device_total_memory(cls, device_id=0): return 16 * 1024**3
    @classmethod
    def pre_register_and_update(cls, parser=None): pass
    @classmethod
    def apply_config_platform_defaults(cls, vllm_config=None): pass
    @classmethod
    def check_and_update_config(cls, vllm_config=None): pass
    @classmethod
    def verify_model_arch(cls, model_arch=None): pass
    @classmethod
    def get_device_uuid(cls, device_id=0): return "vulkan-asahi-0"
    @classmethod
    def supports_fp8(cls): return False
    @classmethod
    def is_fp8_fnuz(cls): return False
    @classmethod
    def fp8_dtype(cls): return torch.float16
    @classmethod
    def supports_mx(cls): return False
    @classmethod
    def get_punica_wrapper(cls): return ""
    @classmethod
    def check_if_supports_dtype(cls, dtype): return True
    @classmethod
    def get_supported_vit_attn_backends(cls):
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
        return [AttentionBackendEnum.TORCH_SDPA]
