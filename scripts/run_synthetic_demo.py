# End-to-end synthetic run (no datasets required)
import os, sys, subprocess
from pathlib import Path
import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # add repo root so 'src' is importable


def main():
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)  # ensure cwd = repo root

    # Make src/ importable for child processes
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)

    # Ensure these exist (harmless if they already do)
    (repo_root/"src/__init__.py").touch()
    (repo_root/"scripts/__init__.py").touch()

    # 1) split
    subprocess.check_call([sys.executable, "-m", "scripts.split_data", "--config", "configs/config.yaml"], env=env)
    # 2) fusion (uses src.* imports)
    subprocess.check_call([sys.executable, "-m", "scripts.run_fusion", "--config", "configs/config.yaml"], env=env)
    # 3) bootstrap CIs
    subprocess.check_call([sys.executable, "-m", "scripts.run_bootstrap_ci", "--config", "configs/config.yaml"], env=env)
    # 4) assemble tables
    subprocess.check_call([sys.executable, "-m", "scripts.reproduce_tables", "--config", "configs/config.yaml"], env=env)
    print("Synthetic demo complete. See out/*.csv and out/*.json")

if __name__ == "__main__":
    main()
