# End-to-end synthetic run (no datasets required)
import os, subprocess, sys
os.makedirs("configs", exist_ok=True)
if not os.path.exists("configs/config.yaml"):
    import shutil
    shutil.copy("configs/config.example.yaml", "configs/config.yaml")
subprocess.check_call([sys.executable, "scripts/split_data.py", "--config", "configs/config.yaml"])
subprocess.check_call([sys.executable, "scripts/run_fusion.py", "--config", "configs/config.yaml"])
subprocess.check_call([sys.executable, "scripts/run_bootstrap_ci.py", "--config", "configs/config.yaml"])
subprocess.check_call([sys.executable, "scripts/reproduce_tables.py", "--config", "configs/config.yaml"])
print("Synthetic demo complete. See out/*.csv and out/*.json")
