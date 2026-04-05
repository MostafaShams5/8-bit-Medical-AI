import pynvml
import time

def monitor_gpu(duration_seconds=60, interval=2):
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    
    if device_count == 0:
        print("No NVIDIA GPU found.")
        return

    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    print("Starting GPU Telemetry...")
    print(f"{'Time':<10} | {'VRAM Used (GB)':<15} | {'GPU Util (%)':<15} | {'Power (W)':<10}")
    
    for _ in range(int(duration_seconds / interval)):
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
        power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        
        vram_used = mem_info.used / (1024**3)
        gpu_util = util_info.gpu
        
        print(f"{time.strftime('%H:%M:%S'):<10} | {vram_used:<15.2f} | {gpu_util:<15} | {power_draw:<10.2f}")
        time.sleep(interval)
        
    pynvml.nvmlShutdown()

if __name__ == "__main__":
    monitor_gpu()
