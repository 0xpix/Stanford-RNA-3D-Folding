from termcolor import colored

def log_message(message, level="INFO"):
    """Write logs to file and print in real-time."""
    formatted_msg = f"[{level}] {message}"
    print(colored(formatted_msg, "cyan" if level == "INFO" else "green" if level == "DONE" else "red"))

def is_gpu(option, params):
    """Check if GPU is enabled and print appropriate message."""
    if "cuda" in params.get("device", ""):
        log_message(f"Using GPU for {option}", "INFO")
    else:
        log_message(f"GPU is not enabled for {option}", "WARNING")