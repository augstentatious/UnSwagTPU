import sys
import time
import shutil
import os

# ANSI Colors
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

def boot_sequence():
    """
    The UnSwag Startup Animation.
    """
    # 1. The Logo
    logo = f"""
{RED}
██╗    ██╗███╗   ██╗███████╗██╗    ██╗ █████╗  ██████╗ 
██║    ██║████╗  ██║██╔════╝██║    ██║██╔══██╗██╔════╝ 
██║    ██║██╔██╗ ██║███████╗██║ █╗ ██║███████║██║  ███╗
██║    ██║██║╚██╗██║╚════██║██║███╗██║██╔══██║██║    ██║
╚██████╔╝██║ ╚████║███████║╚███╔███╔╝██║  ██║╚██████╔╝
 ╚═════╝ ╚═╝  ╚═══╝╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝ ╚═════╝ {RESET}
    """
    print(logo)
    time.sleep(0.2)
    
    # 2. Hardware Recon
    sys.stdout.write(f"{BOLD}[SYSTEM]    {RESET}Scanning silicon... ")
    sys.stdout.flush()
    time.sleep(0.4)
    
    device_type = "Generic CPU"
    count = 1
    backend = "HOST"
    status_color = YELLOW
    
    # --- DUAL STACK SCANNING ---
    found_accelerator = False
    
    # Check JAX (TPU Priority)
    try:
        import jax
        try:
            devices = jax.local_devices()
            platform = devices[0].platform.upper()
            if platform in ["TPU", "GPU"]:
                device_type = platform
                count = len(devices)
                backend = "JAX/XLA"
                status_color = GREEN
                found_accelerator = True
        except:
            pass
    except ImportError:
        pass

    # Check PyTorch (GPU Priority if JAX missed)
    if not found_accelerator:
        try:
            import torch
            if torch.cuda.is_available():
                device_type = torch.cuda.get_device_name(0)
                count = torch.cuda.device_count()
                backend = "TORCH/TRITON"
                status_color = GREEN
                found_accelerator = True
        except ImportError:
            pass

    hw_msg = f"{count}x {device_type} [{backend}]"
    print(f"[{status_color}ONLINE{RESET}]")
    print(f"{BOLD}[HARDWARE]  {RESET}{hw_msg} detected.")
    
    # 3. The Promise
    print(f"{BOLD}[KERNEL]    {RESET}Loading 1-bit Isomorphism...", end="")
    sys.stdout.flush()
    for _ in range(3):
        time.sleep(0.1)
        sys.stdout.write(".")
        sys.stdout.flush()
    print(f" {GREEN}FUSED{RESET}")
    
    # 4. The Flex
    print(f"{BOLD}[MEMORY]    {RESET}Swag Removal Target: {CYAN}96.875%{RESET}")
    print(f"{BOLD}[STATUS]    {RESET}The Memory Wall is now optional.\n")
    
    # 5. Divider
    print(f"{RED}{'—' * 40}{RESET}")

def monitor(iterable, desc="Training"):
    """
    A minimal, UnSwag-styled progress bar.
    """
    total = len(iterable)
    cursor = "🦁" 
    
    print(f"\n{BOLD}{desc} initialized.{RESET}")
    print(f"Offloading gradients to the ether...")
    
    start_time = time.time()
    
    for i, item in enumerate(iterable):
        yield item
        
        if i % 10 == 0 or i == total - 1:
            percent = (i + 1) / total
            bar_len = 30
            filled_len = int(bar_len * percent)
            bar = f"{RED}█{RESET}" * filled_len + f"{CYAN}-{RESET}" * (bar_len - filled_len)
            
            elapsed = time.time() - start_time
            rate = (i + 1) / (elapsed + 1e-9)
            
            sys.stdout.write(f"\r{cursor} [{bar}] {percent:.0%} | {rate:.2f} it/s | {CYAN}1-Bit Mode{RESET}")
            sys.stdout.flush()
            
    print(f"\n{GREEN}Cycle Complete.{RESET}\n")
