import os, sys, torch, types, gc, psutil, contextlib
from vllm.platforms.interface import Platform, PlatformEnum

# --- 1. Memory Identity ---
import vllm.utils
vllm.utils.get_cpu_memory = lambda: 512 * 1024**2
_orig_vmem = psutil.virtual_memory
psutil.virtual_memory = lambda: type("VMem", (), {
    "total": _orig_vmem().total, "available": 512*1024**2, "percent": 99.0,
    "used": _orig_vmem().total - (512*1024**2), "free": 512*1024**2
})()

# --- 2. THE UNIVERSAL ALLOCATOR HIJACK ---
_creation_ops = ['empty', 'zeros', 'ones', 'full', 'arange', 'tensor', 'empty_strided']
_orig_ops = {op: getattr(torch, op) for op in _creation_ops}

def _universal_gaslighter(op_name):
    orig_op = _orig_ops[op_name]
    def wrapper(*args, **kwargs):
        kwargs.pop('pin_memory', None)
        dev = kwargs.get('device')
        if dev is not None and ('cuda' in str(dev).lower() or 'vulkan' in str(dev).lower()):
            kwargs['device'] = 'cpu'
        kwargs.pop('memory_format', None)
        return orig_op(*args, **kwargs)
    return wrapper

for op in _creation_ops:
    setattr(torch, op, _universal_gaslighter(op))

# --- 3. THE OMNI-SYNC HIJACK ---
import torch.cuda

class MockProps:
    def __init__(self):
        self.name = "Asahi Vulkan UMA"; self.major = 8; self.minor = 0
        self.total_memory = 16*1024**3; self.multi_processor_count = 128
        self.gcnArchName = ""; self.is_integrated = 1; self.is_multi_gpu_board = 0

class DSync:
    def __init__(self, *a, **k):
        self.device = torch.device('cpu')
        self.cuda_stream = 0
    def wait_stream(self, *a, **k): pass
    def synchronize(self): pass
    def query(self): return True
    def record(self, *a, **k): pass
    def wait(self, *a, **k): pass
    def elapsed_time(self, other): return 0.0
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def __getattr__(self, n): return lambda *a, **k: None

@contextlib.contextmanager
def _mock_stream_ctx(stream=None):
    yield stream if stream else DSync()

torch.cuda.is_available = lambda: True
torch.cuda.get_device_properties = lambda d=0: MockProps()
torch.cuda.mem_get_info = lambda d=0: (512*1024**2, 16*1024**3)
torch.cuda.get_rng_state = lambda d=0: torch.tensor([0], dtype=torch.uint8)
torch.cuda.current_device = lambda: 0
torch.cuda._lazy_init = lambda: None
torch.cuda.Stream = torch.cuda.Event = torch.Event = DSync # GLOBAL INJECTION
torch.cuda.current_stream = torch.cuda.default_stream = lambda *a, **k: DSync()
torch.cuda.stream = _mock_stream_ctx

# --- 4. Platform Definition ---
class VulkanPlatform(Platform):
    def __init__(self):
        self.device_name = "cpu"; self.device_type = "cpu"; self.dispatch_key = "Vulkan"
    _enum = PlatformEnum.VULKAN
    device_name: str = "vulkan"; device_type: str = "cpu"; dispatch_key: str = "Vulkan"
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
    def is_cpu(self) -> bool: return True
    def is_vulkan(self) -> bool: return True
    @classmethod
    def get_cpu_architecture(cls):
        from vllm.platforms.interface import CpuArchEnum
        return CpuArchEnum.ARM

    # --- 5. Survival Stubs ---
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
    def check_and_update_config(cls, vllm_config=None): pass
    @classmethod
    def fp8_dtype(cls): return torch.float16
    @classmethod
    def get_supported_vit_attn_backends(cls):
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
        return [AttentionBackendEnum.TORCH_SDPA]
