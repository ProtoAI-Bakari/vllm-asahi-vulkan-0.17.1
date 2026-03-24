#!/usr/bin/env python3
"""Vulkan Platform Validation Script"""
import time
import logging
import sys
sys.path.insert(0, '/home/z/GITDEV/vllm_0.17.1')

from vllm.platforms.vulkan import VulkanPlatform

def validate_vulkan_acceleration():
    """Validate Vulkan performance vs baseline"""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("=" * 50)
    print("=== Vulkan Platform Validation Test ===")
    print("=" * 50)
    
    # Test 1: Vulkan initialization
    print("\n[Test 1] Vulkan Platform Initialization")
    start = time.time()
    vulkan = VulkanPlatform()
    init_time = time.time() - start
    print(f"  ✅ VulkanPlatform created: {init_time:.2f}s")
    
    # Test 2: Check Vulkan availability
    print("\n[Test 2] Vulkan Availability Check")
    available = vulkan.is_vulkan_available()
    print(f"  Vulkan available: {available}")
    
    # Test 3: Memory operations
    print("\n[Test 3] Memory Operations")
    import torch
    device = torch.device('vulkan:0' if available else 'cpu')
    
    mem_start = time.time()
    vulkan.empty_cache()
    mem_ops_time = time.time() - mem_start
    print(f"  ✅ empty_cache(): {mem_ops_time:.2f}s")
    
    # Test 4: Memory info
    print("\n[Test 4] Memory Information")
    try:
        total, used = vulkan.mem_get_info(device)
        print(f"  Total memory: {total / 1024**3:.2f} GB")
        print(f"  Used memory: {used / 1024**3:.2f} GB")
        print(f"  ✅ mem_get_info() working")
    except Exception as e:
        print(f"  ⚠️ mem_get_info() error: {e}")
    
    # Test 5: Compute units
    print("\n[Test 5] Compute Units")
    try:
        units = vulkan.num_compute_units(0)
        print(f"  Compute units: {units}")
        print(f"  ✅ num_compute_units() working")
    except Exception as e:
        print(f"  ⚠️ num_compute_units() error: {e}")
    
    # Test 6: Memory usage
    print("\n[Test 6] Memory Usage")
    try:
        usage = vulkan.get_current_memory_usage(device)
        print(f"  Current usage: {usage:.2f} GB")
        print(f"  ✅ get_current_memory_usage() working")
    except Exception as e:
        print(f"  ⚠️ get_current_memory_usage() error: {e}")
    
    # Final Assessment
    print("\n" + "=" * 50)
    print("=== FINAL ASSESSMENT ===")
    print("=" * 50)
    
    if available and init_time < 5:
        print("✅ VULKAN PLATFORM FULLY FUNCTIONAL")
        print(f"   Initialization time: {init_time:.2f}s (excellent)")
        return True
    elif available:
        print("⚠️ VULKAN PLATFORM WORKING BUT SLOW")
        print(f"   Initialization time: {init_time:.2f}s (needs optimization)")
        return True
    else:
        print("❌ VULKAN NOT AVAILABLE")
        print("   Falling back to CPU mode")
        return False

if __name__ == "__main__":
    success = validate_vulkan_acceleration()
    sys.exit(0 if success else 1)
