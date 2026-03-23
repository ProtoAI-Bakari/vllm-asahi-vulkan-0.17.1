import os
import torch
from vllm.platforms.interface import Platform, PlatformEnum

# Force Linear Buffer Layout for FP16 compatibility on Mesa AGX
os.environ["PYTORCH_VULKAN_ENABLE_IMAGE_LAYOUT"] = "0"

class VulkanPlatform(Platform):
    _enum = PlatformEnum.VULKAN
    _device_type = "vulkan"
    _enum = PlatformEnum.VULKAN
    
    # CRITICAL: M1 Max has 32GB Unified Memory (not 16GB)
    _unified_memory_gb = 32.0
    
    def __init__(self): 
        super().__init__()
    
    @classmethod
    def get_instance(cls) -> "VulkanPlatform": 
        return cls()
    
    @property
    def device_type(self) -> str: 
        return self._device_type
    
    def is_vulkan_available(self) -> bool: 
        return torch.is_vulkan_available()
    
    @classmethod
    def is_pin_memory_available(cls) -> bool:
        """Vulkan does not support pinned memory like CUDA."""
        return False
    
    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend,
        attn_selector_config,
        num_heads=None,
    ) -> str:
        """Vulkan uses Torch SDPA attention backend since it's pure PyTorch."""
        return "vllm.v1.attention.backends.flex_attention.FlexAttentionBackend"
    
    @classmethod
    def get_worker_cls(cls, vllm_config, local_rank: int, rank: int, distributed_init_method: str, is_driver_worker: bool = False):
        """Force GPU Worker for Vulkan platform to enable GPU acceleration."""
        return "vllm.v1.worker.gpu_worker.Worker"
    
    @classmethod
    def get_model_runner_cls(cls, vllm_config):
        """Force GPUModelRunner for Vulkan platform to enable GPU acceleration."""
        return "vllm.v1.worker.gpu_model_runner.GPUModelRunner"
    
    def set_device(self, device: torch.device) -> None:
        """Set the Vulkan device for computation."""
        self._device = device
    
    def check_if_supports_dtype(self, dtype: torch.dtype) -> None:
        """Check if Vulkan supports the given dtype."""
        if dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(f"Vulkan does not support dtype {dtype}")
    
    def memory_stats(self, device: torch.device) -> dict:
        """Return memory stats for Vulkan device."""
        stats = {}
        try:
            if torch.is_vulkan_available():
                # Try to get memory info from PyTorch
                free, total = self.mem_get_info(device)
                stats['total'] = total
                stats['free'] = free
                stats['used'] = total - free
        except:
            pass
        return stats
    
    def mem_get_info(self, device: torch.device) -> tuple[int, int]:
        """Get memory info for Vulkan device.
        
        Returns:
            tuple of (free_memory, total_memory) in bytes
        """
        try:
            if torch.is_vulkan_available():
                # PyTorch Vulkan doesn't expose direct memory info like CUDA
                # Use our known hardware specs (M1 Max = 32GB unified)
                total = self._unified_memory_gb * 1024 * 1024 * 1024  # 32GB in bytes
                
                # Try to estimate used memory
                if hasattr(torch, 'vulkan'):
                    if hasattr(torch.vulkan, 'memory_stats'):
                        stats = torch.vulkan.memory_stats(device)
                        used = stats.get('allocated', 0)
                        free = total - used
                        return (free, total)
                
                # Fallback: return full memory (conservative estimate)
                return (total, total)
        except Exception as e:
            # If all else fails, return our hardware spec
            total = self._unified_memory_gb * 1024 * 1024 * 1024
            return (total, total)
        
        # Ultimate fallback: use our known hardware
        total = self._unified_memory_gb * 1024 * 1024 * 1024
        return (total, total)
    
    def memory_reserved(self, device: torch.device) -> int:
        """Get reserved memory for Vulkan device."""
        try:
            if torch.is_vulkan_available():
                # Try to get reserved memory
                if hasattr(torch, 'vulkan') and hasattr(torch.vulkan, 'memory_reserved'):
                    return torch.vulkan.memory_reserved(device)
        except:
            pass
        # Return conservative estimate (10% of total)
        return int(self._unified_memory_gb * 1024 * 1024 * 1024 * 0.1)
    
    def num_compute_units(self, device_id: int = 0) -> int:
        """Return number of compute units for Vulkan device.
        M1 Max has ~4096 compute units (based on GPU architecture)
        """
        return 4096
    
    def get_current_memory_usage(self, device: torch.device) -> float:
        """Return current memory usage for Vulkan device in GB."""
        try:
            if torch.is_vulkan_available():
                # Try to get memory usage from PyTorch
                if hasattr(torch, 'vulkan'):
                    if hasattr(torch.vulkan, 'memory_stats'):
                        stats = torch.vulkan.memory_stats(device)
                        allocated = stats.get('allocated', 0)
                        return allocated / (1024 ** 3)  # Convert to GB
                
                # Fallback: estimate based on process memory
                try:
                    import psutil
                    process = psutil.Process()
                    return round(process.memory_info().rss / (1024 ** 3), 2)
                except:
                    pass
        except:
            pass
        return 0.0
    
    def memory_available(self) -> float:
        """Return available GPU memory in GB."""
        return self._unified_memory_gb
    
    def get_device_capabilities(self) -> dict:
        """Return device capabilities dict."""
        return {
            'memory_gb': self._unified_memory_gb,
            'compute_units': 4096,
            'vulkan_version': '1.3',
            'unified_memory': True,
            'device_type': 'vulkan',
            'platform': 'Apple M1 Max'
        }
    
    def is_acceleration_available(self) -> bool:
        """Direct check if Vulkan acceleration is working."""
        return self.is_vulkan_available()
    
    def empty_cache(self):
        """Empty cache for Vulkan device."""
        # For Vulkan, try to use torch memory clearing
        try:
            if hasattr(torch, 'vulkan') and torch.vulkan.is_available():
                torch.vulkan.empty_cache()
            else:
                # Fallback to CPU garbage collection
                import gc
                gc.collect()
        except:
            pass  # Ignore errors
    
    def reset_peak_memory_stats(self, device: torch.device):
        """Reset peak memory stats for Vulkan device."""
        # For Vulkan, no-op since we don't track peak stats
        pass

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        """Set default block_size for Vulkan platform."""
        import sys
        print(f"DEBUG: check_and_update_config called", file=sys.stderr, flush=True)
        cache_config = vllm_config.cache_config
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 16
            print(f"DEBUG: Set block_size to 16", file=sys.stderr, flush=True)
        
        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm.v1.worker.gpu_worker.Worker"
            print(f"DEBUG: Set worker_cls to gpu_worker.Worker", file=sys.stderr, flush=True)
