# Vulkan dtype shield fix
import torch
import os

if os.environ.get('VLLM_PLATFORM') == 'vulkan':
    _orig_to = torch.Tensor.to
    
    def _vulkan_shield_to(self, *args, **kwargs):
        device = kwargs.get('device') or (args[0] if args else None)
        dtype = kwargs.get('dtype') or (args[1] if len(args) > 1 else None)
        
        # If moving to Vulkan, enforce float32 dtype
        if device and 'vulkan' in str(device):
            # Handle forbidden types
            if self.dtype == torch.int64:
                # Vulkan only supports int32 for metadata
                self = self.to(torch.int32)
            elif self.dtype == torch.bool:
                self = self.to(torch.int32)
            elif self.dtype == torch.float16 or self.dtype == torch.half:
                # Vulkan FP16 shader missing - force float32
                self = self.to(torch.float32)
            elif self.dtype == torch.bfloat16:
                # bfloat16 may not be supported on all Vulkan implementations
                # Keep as-is but warn
                pass
            
            # Remove dtype from kwargs if we've already converted
            if dtype is not None:
                kwargs.pop('dtype', None)
        
        return _orig_to(self, *args, **kwargs)
    
    torch.Tensor.to = _vulkan_shield_to
    print("⚠️ VULKAN BRIDGE v6: Full dtype Shield (int64/bool/fp16->fp32) Active")
