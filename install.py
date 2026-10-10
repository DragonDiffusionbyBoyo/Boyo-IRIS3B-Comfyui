"""
ComfyUI Manager calls this automatically on first install.
Installs the vendored iris-3b package into the ComfyUI Python environment.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def install_requirements():
    req = HERE / "requirements.txt"
    if req.exists():
        print("[Iris3B] Installing requirements...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(req)
        ])


def install_iris3b():
    iris_path = HERE / "iris-3b"
    if not iris_path.exists():
        print("[Iris3B] ERROR: iris-3b directory not found in node pack.")
        print("[Iris3B] The vendored iris-3b folder should be present in the repo.")
        return False

    pyproject = iris_path / "pyproject.toml"
    if not pyproject.exists():
        print(f"[Iris3B] ERROR: no pyproject.toml found in {iris_path}")
        return False

    print(f"[Iris3B] Installing iris-3b package from {iris_path} ...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-e", str(iris_path)
        ])
        print("[Iris3B] iris-3b installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Iris3B] ERROR: pip install failed: {e}")
        print("[Iris3B] Try running manually:")
        print(f"    pip install -e \"{iris_path}\"")
        return False


if __name__ == "__main__":
    print("[Iris3B] Running installer...")
    install_requirements()
    ok = install_iris3b()
    if ok:
        print("[Iris3B] Installation complete.")
        print("[Iris3B] Restart ComfyUI, then use the 'Iris-3B Download Models' node to fetch weights.")
    else:
        print("[Iris3B] Installation completed with errors — see above.")
