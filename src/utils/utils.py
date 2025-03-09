from termcolor import colored

import jax

def log_message(message, level="INFO"):
    """Write logs to file and print in real-time with distinct colors."""
    colors = {
        "INFO": "cyan",     # ℹ️ Cyan for informational messages
        "DONE": "green",    # ✅ Green for successful completion
        "ERROR": "red",     # ❌ Red for errors
        "WARN": "yellow",   # ⚠️ Yellow for warnings
        "PASS": "blue",     # 🟦 Blue for passed tests
        "FAIL": "magenta"   # 🟪 Magenta for failed tests
    }

    formatted_msg = f"[{level}] {message}"
    print(colored(formatted_msg, colors.get(level, "white")))  # Default to white if level is unknown



# utils.py

def check_jax_device():
    """
    Checks if JAX is using a GPU or CPU and prints device details.
    """
    devices = jax.devices()

    if any("cuda" in str(d).lower() or "gpu" in str(d).lower() for d in devices):
        log_message("✅ JAX is using the GPU")
    else:
        log_message("⚠️ JAX is running on the CPU. Install CUDA-enabled JAX for GPU support.")