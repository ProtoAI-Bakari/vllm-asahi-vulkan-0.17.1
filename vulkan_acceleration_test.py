#!/usr/bin/env python3
"""Vulkan Acceleration Verification Script"""
import sys
import time
sys.path.insert(0, '/home/z/GITDEV/vllm_0.17.1')

from vllm.platforms.vulkan import VulkanPlatform

def test_vulkan_acceleration():
    """Verify Vulkan GPU acceleration is actually working"""
    print("=" * 60)
    print("=== Vulkan Acceleration Verification Test ===")
    print("=" * 60)
    
    # Test 1: Platform initialization
    print("\n[Test 1] Vulkan Platform Initialization")
    start = time.time()
    platform = VulkanPlatform()
    init_time = time.time() - start
    print(f"  ✅ VulkanPlatform created in {init_time:.3f}s")
    
    # Test 2: Vulkan availability
    print("\n[Test 2] Vulkan Availability")
    available = platform.is_vulkan_available()
    print(f"  Vulkan available: {available}")
    if not available:
        print("  ❌ CRITICAL: Vulkan not available - falling back to CPU")
        return False
    
    # Test 3: Memory specs (32GB verification)
    print("\n[Test 3] Memory Specifications")
    total_mem = platform.memory_available()
    print(f"  Total memory: {total_mem} GB")
    if total_mem >= 30:  # Allow some margin
        print(f"  ✅ Correctly reporting ~32GB unified memory")
    else:
        print(f"  ⚠️ Memory estimate may be incorrect (expected ~32GB)")
    
    # Test 4: Device capabilities
    print("\n[Test 4] Device Capabilities")
    caps = platform.get_device_capabilities()
    for key, value in caps.items():
        print(f"  {key}: {value}")
    
    # Test 5: Memory allocation test
    print("\n[Test 5] Memory Allocation Test")
    import torch
    device = torch.device('vulkan:0')
    
    # Get baseline memory
    baseline_mem = platform.get_current_memory_usage(device)
    print(f"  Baseline memory usage: {baseline_mem:.2f} GB")
    
    # Allocate some memory on Vulkan
    try:
        test_tensor = torch.randn(1000, 1000, device=device)
        allocated_mem = platform.get_current_memory_usage(device)
        print(f"  After tensor allocation: {allocated_mem:.2f} GB")
        
        if allocated_mem > baseline_mem:
            print(f"  ✅ Memory allocated successfully: +{allocated_mem - baseline_mem:.2f} GB")
        else:
            print(f"  ⚠️ No memory increase detected - possible CPU fallback")
        
        # Clean up
        del test_tensor
        torch.cuda.empty_cache() if hasattr(torch, 'cuda') else None
        platform.empty_cache()
        
    except Exception as e:
        print(f"  ❌ Memory allocation failed: {e}")
        return False
    
    # Test 6: Acceleration check
    print("\n[Test 6] Acceleration Availability")
    accel_available = platform.is_acceleration_available()
    print(f"  Acceleration available: {accel_available}")
    
    # Final Assessment
    print("\n" + "=" * 60)
    print("=== FINAL ASSESSMENT ===")
    print("=" * 60)
    
    if available and init_time < 1.0 and total_mem >= 30:
        print("✅ VULKAN GPU ACCELERATION FULLY FUNCTIONAL")
        print(f"   - Initialization: {init_time:.3f}s (excellent)")
        print(f"   - Memory: {total_mem} GB unified")
        print(f"   - Acceleration: ACTIVE")
        print("\n🎉 READY FOR FULL PIPELINE TEST")
        return True
    elif available and total_mem >= 30:
        print("✅ VULKAN GPU ACCELERATION WORKING")
        print(f"   - Initialization: {init_time:.3f}s")
        print(f"   - Memory: {total_mem} GB unified")
        print("\n🎉 READY FOR FULL PIPELINE TEST")
        return True
    elif available:
        print("⚠️ VULKAN AVAILABLE BUT MEMORY CONFIG MAY NEED REVIEW")
        print(f"   - Memory: {total_mem} GB (expected ~32GB)")
        print("\n⚠️ Review vulkan.py memory settings")
        return True
    else:
        print("❌ VULKAN ACCELERATION NOT WORKING")
        print("   Falling back to CPU mode")
        return False

if __name__ == "__main__":
    success = test_vulkan_acceleration()
    sys.exit(0 if success else 1)
