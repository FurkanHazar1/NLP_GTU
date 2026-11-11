#!/usr/bin/env python3
"""Environment verification script for Team 07 - TOXICITY.

Project: Explainable Turkish Implicit Toxicity Detection.
Validates Python runtime, core dependencies, PyTorch, and GPU availability.
"""

from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version


def get_pkg_version(package_name: str) -> str:
    """Retrieve installed version of a package."""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "NOT INSTALLED"


def print_section(title: str) -> None:
    """Print formatted section header."""
    print(f"\n{'=' * 70}")
    print(f" {title}")
    print(f"{'=' * 70}")


def check_runtime() -> dict[str, str]:
    """Inspect Python runtime and platform."""
    print_section("1. Runtime Environment")
    py_ver = sys.version.split()[0]
    os_info = f"{platform.system()} {platform.release()} ({platform.machine()})"
    print(f"  Python Version:     {py_ver}")
    print(f"  Python Executable:  {sys.executable}")
    print(f"  OS / Architecture:  {os_info}")

    major, minor, _ = sys.version_info[:3]
    if not (major == 3 and 11 <= minor < 13):
        print(f"  [WARN] Python version {py_ver} outside recommended range >=3.11, <3.13")
    else:
        print("  [OK] Python interpreter satisfies >=3.11, <3.13")

    return {"Python": py_ver, "OS": os_info}


def check_packages() -> dict[str, str]:
    """Check core packages and versions."""
    print_section("2. Package Verification")

    required_packages = [
        # Scientific computing
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("scipy", "SciPy"),
        # Classical ML
        ("scikit-learn", "scikit-learn"),
        ("joblib", "joblib"),
        # Deep learning
        ("torch", "PyTorch"),
        ("accelerate", "Accelerate"),
        # NLP
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("evaluate", "Evaluate"),
        ("huggingface_hub", "Hugging Face Hub"),
        ("safetensors", "Safetensors"),
        # Explainable AI
        ("shap", "SHAP"),
        # Visualization
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        # Utilities
        ("tqdm", "tqdm"),
        ("pyyaml", "PyYAML"),
        ("psutil", "psutil"),
        # Dev & Data tools
        ("pytest", "pytest"),
        ("ruff", "Ruff"),
        ("jupyterlab", "JupyterLab"),
        ("kagglehub", "kagglehub"),
    ]

    results: dict[str, str] = {}
    missing: list[str] = []

    print(f"  {'Package':<22} {'Installed Version':<20} {'Status'}")
    print(f"  {'-' * 20:<22} {'-' * 18:<20} {'-' * 8}")

    for pkg_id, display_name in required_packages:
        ver = get_pkg_version(pkg_id)
        results[pkg_id] = ver
        if ver == "NOT INSTALLED":
            status = "[FAILED]"
            missing.append(display_name)
        else:
            status = "[OK]"
        print(f"  {display_name:<22} {ver:<20} {status}")

    if missing:
        print(f"\n  [ERROR] Missing packages: {', '.join(missing)}")
    else:
        print("\n  [OK] All core and development packages are installed.")

    return results


def check_imports_and_functionality() -> bool:
    """Verify library imports and basic functionality without external downloads."""
    print_section("3. Library Import & Subsystem Verification")
    all_ok = True

    # 1. Scientific computing
    try:
        import numpy as np
        import pandas as pd
        import scipy  # noqa: F401

        arr = np.array([1.0, 2.0, 3.0])
        df = pd.DataFrame({"col": arr})
        assert len(df) == 3
        print("  [OK] NumPy, Pandas, SciPy imported and verified")
    except (ImportError, AssertionError, RuntimeError) as exc:
        print(f"  [FAIL] Scientific stack error: {exc}")
        all_ok = False

    # 2. Classical Machine Learning
    try:
        import joblib  # noqa: F401
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression()
        clf.fit([[0, 0], [1, 1]], [0, 1])
        assert clf.predict([[1, 1]])[0] == 1
        print("  [OK] scikit-learn & joblib imported and verified")
    except (ImportError, AssertionError, RuntimeError) as exc:
        print(f"  [FAIL] scikit-learn error: {exc}")
        all_ok = False

    # 3. Deep Learning (PyTorch & Accelerate)
    try:
        import accelerate  # noqa: F401
        import torch

        t = torch.tensor([1.0, 2.0])
        assert t.shape == (2,)
        print(f"  [OK] PyTorch ({torch.__version__}) & Accelerate imported")
    except (ImportError, AssertionError, RuntimeError) as exc:
        print(f"  [FAIL] PyTorch / Accelerate error: {exc}")
        all_ok = False

    # 4. GPU & CUDA Verification
    try:
        import torch

        cuda_avail = torch.cuda.is_available()
        print(f"  PyTorch CUDA Available: {cuda_avail}")
        if cuda_avail:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            torch_cuda_ver = torch.version.cuda
            print(f"  CUDA Version (Torch):   {torch_cuda_ver}")
            print(f"  GPU Count:              {gpu_count}")
            print(f"  Primary GPU Model:      {gpu_name}")
            x = torch.zeros((1,), device="cuda")
            assert x.is_cuda
            print("  [OK] CUDA device operation verified successfully")
        else:
            print("  [INFO] Running on CPU (no CUDA GPU detected or available)")
    except (ImportError, AssertionError, RuntimeError) as exc:
        print(f"  [FAIL] CUDA verification error: {exc}")
        all_ok = False

    # 5. NLP Libraries (Transformers, Datasets, Evaluate, Safetensors)
    try:
        import datasets  # noqa: F401
        import evaluate  # noqa: F401
        import huggingface_hub  # noqa: F401
        import safetensors  # noqa: F401
        from transformers import BertConfig

        config = BertConfig(vocab_size=32000, hidden_size=768)
        assert config.hidden_size == 768
        print("  [OK] Transformers, Datasets, Evaluate, Safetensors imported")
    except (ImportError, AssertionError, RuntimeError) as exc:
        print(f"  [FAIL] NLP stack error: {exc}")
        all_ok = False

    # 6. Explainable AI (SHAP)
    try:
        import shap

        assert hasattr(shap, "Explainer")
        print(f"  [OK] SHAP ({shap.__version__}) imported successfully")
    except (ImportError, AssertionError, RuntimeError) as exc:
        print(f"  [FAIL] SHAP import error: {exc}")
        all_ok = False

    # 7. Visualization (Matplotlib & Seaborn)
    try:
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt  # noqa: F401
        import seaborn as sns  # noqa: F401

        print("  [OK] Matplotlib & Seaborn imported (Agg backend)")
    except (ImportError, AssertionError, RuntimeError) as exc:
        print(f"  [FAIL] Visualization stack error: {exc}")
        all_ok = False

    return all_ok


def main() -> int:
    """Run all validation checks and print summary."""
    print("=" * 70)
    print(" Team 07 - TOXICITY | Environment Verification Report")
    print(" Project: Explainable Turkish Implicit Toxicity Detection")
    print("=" * 70)

    check_runtime()
    check_packages()
    success = check_imports_and_functionality()

    print_section("Summary")
    if success:
        print("  >>> Environment validation PASSED successfully! <<<")
        print("  All required libraries, PyTorch, and GPU subsystems are operational.\n")
        return 0
    else:
        print("  >>> Environment validation FAILED with errors. <<<\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
