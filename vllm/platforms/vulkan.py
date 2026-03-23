import os
import torch
from vllm.platforms.interface import Platform

# Force Linear Buffer Layout for FP16 compatibility on Mesa AGX
os.environ["PYTORCH_VULKAN_ENABLE_IMAGE_LAYOUT"] = "0"

class VulkanPlatform(Platform):
    _device_type = "vulkan"
    def __init__(self): super().__init__()
    @classmethod
    def get_instance(cls) -> "VulkanPlatform": return cls()
    @property
    def device_type(self) -> str: return self._device_type
    def is_vulkan_available(self) -> bool: return torch.is_vulkan_available()
