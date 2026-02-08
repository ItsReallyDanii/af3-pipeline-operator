#!/usr/bin/env python3
"""collect_diagnostics.py — Gather environment diagnostics into a bundle.

Collects:
  - Environment/package versions
  - nvidia-smi output
  - nvcc --version output
  - PyTorch CUDA report
  - Latest run artifacts from known output directories
  - Conda/pip freeze state

Outputs:
  - diagnostics_bundle.json  (machine-readable)
  - diagnostics_bundle.md    (human-readable)

Usage:
    python scripts/collect_diagnostics.py
    python scripts/collect_diagnostics.py --output-dir /path/to/dir
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run(cmd: list[str], timeout: int = 15) -> dict:
    """Run command and return structured result."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {
            "command": " ".join(cmd),
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except FileNotFoundError:
        return {"command": " ".join(cmd), "returncode": -1, "stdout": "", "stderr": f"{cmd[0]} not found"}
    except subprocess.TimeoutExpired:
        return {"command": " ".join(cmd), "returncode": -2, "stdout": "", "stderr": "timeout"}
    except Exception as exc:
        return {"command": " ".join(cmd), "returncode": -3, "stdout": "", "stderr": str(exc)}


def _get_version(module_name: str, import_name: str | None = None) -> str:
    """Get version of a Python package, return 'not installed' on failure."""
    try:
        mod = importlib.import_module(import_name or module_name)
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return "not installed"


# ── Collectors ────────────────────────────────────────────────────────────────

def collect_system_info() -> dict:
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": platform.node(),
        "os_release": platform.release(),
        "cpu_count": os.cpu_count(),
        "cwd": os.getcwd(),
        "conda_prefix": os.environ.get("CONDA_PREFIX", "not set"),
        "cuda_home": os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH", "not set")),
        "virtual_env": os.environ.get("VIRTUAL_ENV", "not set"),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", "not set"),
    }


def collect_gpu_info() -> dict:
    nvidia_smi = _run(["nvidia-smi"])
    nvidia_smi_query = _run([
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total,memory.free,temperature.gpu",
        "--format=csv,noheader"
    ])
    nvcc = _run(["nvcc", "--version"])
    return {
        "nvidia_smi": nvidia_smi,
        "nvidia_smi_query": nvidia_smi_query,
        "nvcc_version": nvcc,
    }


def collect_torch_cuda_report() -> dict:
    result = {"available": False}
    try:
        import torch
        result["torch_version"] = torch.__version__
        result["torch_cuda_version"] = torch.version.cuda or "N/A"
        result["torch_cudnn_version"] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A"
        result["cuda_available"] = torch.cuda.is_available()
        result["available"] = True
        if torch.cuda.is_available():
            result["device_count"] = torch.cuda.device_count()
            result["current_device"] = torch.cuda.current_device()
            result["device_name"] = torch.cuda.get_device_name(0)
            prop = torch.cuda.get_device_properties(0)
            result["device_capability"] = f"{prop.major}.{prop.minor}"
            result["total_memory_mb"] = round(prop.total_mem / 1024 / 1024, 1)
    except ImportError:
        result["error"] = "torch not installed"
    except Exception as exc:
        result["error"] = str(exc)
    return result


def collect_package_versions() -> dict:
    packages = [
        ("torch", None),
        ("deepspeed", None),
        ("pytorch_lightning", None),
        ("torchmetrics", None),
        ("numpy", None),
        ("scipy", None),
        ("pandas", None),
        ("pydantic", None),
        ("pyyaml", "yaml"),
        ("requests", None),
        ("click", None),
        ("biopython", "Bio"),
        ("h5py", None),
        ("numba", None),
        ("datasets", None),
        ("fsspec", None),
        ("s3fs", None),
        ("rich", None),
        ("triton", None),
        ("certifi", None),
    ]
    return {name: _get_version(name, imp) for name, imp in packages}


def collect_conda_state() -> dict:
    conda_list = _run(["conda", "list", "--json"])
    conda_info = _run(["conda", "info", "--json"])
    return {
        "conda_list": conda_list,
        "conda_info": conda_info,
    }


def collect_pip_state() -> dict:
    pip_freeze = _run([sys.executable, "-m", "pip", "freeze"])
    return {"pip_freeze": pip_freeze}


def collect_run_artifacts() -> dict:
    """Scan known output directories for latest run artifacts."""
    repo_root = Path(__file__).resolve().parent.parent
    scan_dirs = [
        repo_root / "runs",
        Path("/tmp/of3test"),
    ]

    artifacts = {}
    for scan_dir in scan_dirs:
        dir_key = str(scan_dir)
        if not scan_dir.is_dir():
            artifacts[dir_key] = {"exists": False}
            continue

        entries = []
        for child in sorted(scan_dir.iterdir()):
            if child.is_dir():
                manifest = child / "manifest.json"
                entry = {
                    "name": child.name,
                    "has_manifest": manifest.is_file(),
                }
                if manifest.is_file():
                    try:
                        data = json.loads(manifest.read_text(encoding="utf-8"))
                        entry["status"] = data.get("status", "unknown")
                        entry["run_id"] = data.get("run_id", "unknown")
                        entry["timestamp"] = data.get("timestamp_start_utc", "unknown")
                    except (json.JSONDecodeError, OSError):
                        entry["status"] = "manifest_unreadable"

                # Check for log files
                log_dir = child / "logs"
                if log_dir.is_dir():
                    log_files = list(log_dir.glob("*.log"))
                    entry["log_files"] = [f.name for f in log_files]
                else:
                    entry["log_files"] = []

                # Check for output files
                output_files = list(child.glob("*.pdb")) + list(child.glob("*.cif"))
                entry["output_files"] = [f.name for f in output_files]

                entries.append(entry)

        artifacts[dir_key] = {"exists": True, "runs": entries}

    return artifacts


# ── Output Formatters ─────────────────────────────────────────────────────────

def bundle_to_markdown(bundle: dict) -> str:
    lines = [
        "# OpenFold3 + DeepSpeed — Diagnostics Bundle",
        f"",
        f"**Generated:** {bundle['timestamp']}",
        f"",
        "---",
        "",
        "## System Info",
        "",
    ]
    for k, v in bundle["system"].items():
        lines.append(f"- **{k}:** `{v}`")

    lines += ["", "## GPU Info", ""]
    nv = bundle["gpu"]["nvidia_smi_query"]
    lines.append(f"```\n{nv.get('stdout', nv.get('stderr', 'N/A'))}\n```")

    nvcc = bundle["gpu"]["nvcc_version"]
    lines.append(f"\n**nvcc:**\n```\n{nvcc.get('stdout', nvcc.get('stderr', 'N/A'))}\n```")

    lines += ["", "## PyTorch CUDA Report", ""]
    for k, v in bundle["torch_cuda"].items():
        lines.append(f"- **{k}:** `{v}`")

    lines += ["", "## Package Versions", ""]
    lines.append("| Package | Version |")
    lines.append("|---------|---------|")
    for k, v in bundle["packages"].items():
        lines.append(f"| {k} | {v} |")

    lines += ["", "## Run Artifacts", ""]
    for dir_path, info in bundle["artifacts"].items():
        lines.append(f"### `{dir_path}`")
        if not info.get("exists"):
            lines.append("Directory does not exist.\n")
            continue
        runs = info.get("runs", [])
        if not runs:
            lines.append("No run directories found.\n")
            continue
        for run in runs:
            status = run.get("status", "no manifest")
            lines.append(f"- **{run['name']}**: status=`{status}`, "
                         f"outputs={run.get('output_files', [])}, "
                         f"logs={run.get('log_files', [])}")
        lines.append("")

    lines += ["", "## Pip Freeze", "", "```"]
    pip_out = bundle.get("pip", {}).get("pip_freeze", {}).get("stdout", "N/A")
    lines.append(pip_out)
    lines.append("```")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Collect environment diagnostics")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory to write diagnostic bundles")
    args = parser.parse_args()

    print("Collecting diagnostics...")

    bundle = {
        "timestamp": datetime.now().isoformat(),
        "system": collect_system_info(),
        "gpu": collect_gpu_info(),
        "torch_cuda": collect_torch_cuda_report(),
        "packages": collect_package_versions(),
        "conda": collect_conda_state(),
        "pip": collect_pip_state(),
        "artifacts": collect_run_artifacts(),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "diagnostics_bundle.json"
    json_path.write_text(json.dumps(bundle, indent=2, default=str), encoding="utf-8")
    print(f"  -> {json_path}")

    md_path = out_dir / "diagnostics_bundle.md"
    md_path.write_text(bundle_to_markdown(bundle), encoding="utf-8")
    print(f"  -> {md_path}")

    print("Done.")


if __name__ == "__main__":
    main()
