#!/usr/bin/env python3
"""preflight.py — Environment verification for OpenFold3 + DeepSpeed.

Performs ONLY checks. No model inference. No simulations. No mutations.

Exit codes:
    0  All checks passed
    1  One or more checks failed (details printed)

Usage:
    python scripts/preflight.py
    python scripts/preflight.py --json          # machine-readable output
    python scripts/preflight.py --output report.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import socket
import ssl
import struct
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ── Result accumulator ────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    severity: str = "error"  # "error" | "warning" | "info"


@dataclass
class PreflightReport:
    python_version: str = ""
    platform_info: str = ""
    checks: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str, severity: str = "error") -> None:
        self.checks.append(CheckResult(name=name, passed=passed, detail=detail, severity=severity))

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks if c.severity == "error")

    def summary(self) -> dict:
        return {
            "python_version": self.python_version,
            "platform": self.platform_info,
            "total_checks": len(self.checks),
            "passed": sum(1 for c in self.checks if c.passed),
            "failed": sum(1 for c in self.checks if not c.passed and c.severity == "error"),
            "warnings": sum(1 for c in self.checks if not c.passed and c.severity == "warning"),
            "overall": "PASS" if self.all_passed else "FAIL",
            "checks": [asdict(c) for c in self.checks],
        }


report = PreflightReport()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_cmd(cmd: list[str], timeout: int = 10) -> tuple[int, str]:
    """Run a command, return (returncode, stdout+stderr). Never raises."""
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return proc.returncode, (proc.stdout + proc.stderr).strip()
    except FileNotFoundError:
        return -1, f"Command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return -2, f"Command timed out after {timeout}s"
    except Exception as exc:
        return -3, str(exc)


def _header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _check(label: str, passed: bool, detail: str, severity: str = "error") -> None:
    icon = "PASS" if passed else ("WARN" if severity == "warning" else "FAIL")
    print(f"  [{icon}] {label}: {detail}")
    report.add(label, passed, detail, severity)


# ── 1. Python / OS / Architecture ────────────────────────────────────────────

def check_python_os() -> None:
    _header("Python / OS / Architecture")

    py_ver = platform.python_version()
    report.python_version = py_ver
    report.platform_info = platform.platform()

    _check("Python version", py_ver.startswith("3.12"), f"{py_ver} (need 3.12.x)")
    _check("OS", sys.platform == "linux", f"{sys.platform} (need linux)")
    _check("Architecture", platform.machine() == "x86_64",
           f"{platform.machine()} (need x86_64)")
    _check("Pointer size", struct.calcsize("P") * 8 == 64, f"{struct.calcsize('P')*8}-bit")
    _check("Platform detail", True, report.platform_info, severity="info")


# ── 2. GPU / CUDA Visibility ─────────────────────────────────────────────────

def check_gpu_cuda() -> None:
    _header("GPU / CUDA Visibility")

    # nvidia-smi
    rc, out = _run_cmd(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                        "--format=csv,noheader,nounits"])
    if rc == 0:
        _check("nvidia-smi", True, out.split("\n")[0])
    else:
        _check("nvidia-smi", False, out)

    # nvcc
    rc, out = _run_cmd(["nvcc", "--version"])
    if rc == 0:
        ver_line = [l for l in out.split("\n") if "release" in l.lower()]
        _check("nvcc", True, ver_line[0].strip() if ver_line else out[:80])
    else:
        _check("nvcc", False, f"nvcc not found or failed: {out[:120]}")

    # CUDA_HOME / CUDA_PATH
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH", "")
    if cuda_home and Path(cuda_home).is_dir():
        _check("CUDA_HOME", True, cuda_home)
    else:
        # Try to infer from nvcc location
        nvcc_path = shutil.which("nvcc")
        if nvcc_path:
            inferred = str(Path(nvcc_path).resolve().parent.parent)
            _check("CUDA_HOME", True, f"inferred from nvcc: {inferred}", severity="warning")
        else:
            _check("CUDA_HOME", False, "CUDA_HOME/CUDA_PATH not set and nvcc not on PATH")


# ── 3. CUDA Headers / Libraries ──────────────────────────────────────────────

def check_cuda_headers_libs() -> None:
    _header("CUDA Headers & Libraries")

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH", "")
    if not cuda_home:
        nvcc_path = shutil.which("nvcc")
        if nvcc_path:
            cuda_home = str(Path(nvcc_path).resolve().parent.parent)

    # Headers to check
    required_headers = ["cusolverDn.h", "cusparse.h", "cublas_v2.h"]
    search_dirs = []
    if cuda_home:
        search_dirs.append(Path(cuda_home) / "include")
    # Also check conda env
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        search_dirs.append(Path(conda_prefix) / "include")
        search_dirs.append(Path(conda_prefix) / "targets" / "x86_64-linux" / "include")

    for header in required_headers:
        found = False
        found_path = ""
        for d in search_dirs:
            candidate = d / header
            if candidate.is_file():
                found = True
                found_path = str(candidate)
                break
        _check(f"Header: {header}", found,
               found_path if found else f"not found in {[str(d) for d in search_dirs]}")

    # Shared libraries to check
    required_libs = ["libcublas.so", "libcusolver.so", "libcusparse.so"]
    lib_dirs = []
    if cuda_home:
        lib_dirs.append(Path(cuda_home) / "lib64")
        lib_dirs.append(Path(cuda_home) / "lib")
    if conda_prefix:
        lib_dirs.append(Path(conda_prefix) / "lib")
        lib_dirs.append(Path(conda_prefix) / "targets" / "x86_64-linux" / "lib")

    for lib in required_libs:
        found = False
        found_path = ""
        for d in lib_dirs:
            # Check for exact name or versioned variants
            if d.is_dir():
                matches = list(d.glob(f"{lib}*"))
                if matches:
                    found = True
                    found_path = str(matches[0])
                    break
        _check(f"Library: {lib}", found,
               found_path if found else f"not found in {[str(d) for d in lib_dirs]}")


# ── 4. PyTorch / DeepSpeed Compatibility ─────────────────────────────────────

def check_torch_deepspeed() -> None:
    _header("PyTorch / DeepSpeed Compatibility")

    # torch
    try:
        import torch
        _check("torch import", True, f"v{torch.__version__}")
        _check("torch.cuda.is_available()", torch.cuda.is_available(),
               str(torch.cuda.is_available()))
        if torch.cuda.is_available():
            _check("torch.cuda.device_count()", torch.cuda.device_count() >= 1,
                   str(torch.cuda.device_count()))
            _check("torch CUDA version", True, torch.version.cuda or "N/A", severity="info")
            _check("GPU name", True, torch.cuda.get_device_name(0), severity="info")
            _check("cuDNN version", True,
                   str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A",
                   severity="info")
        else:
            _check("torch CUDA", False, "CUDA not available in torch")
    except ImportError as exc:
        _check("torch import", False, str(exc))

    # deepspeed
    try:
        import deepspeed
        _check("deepspeed import", True, f"v{deepspeed.__version__}")

        # Check deepspeed ops compatibility
        rc, out = _run_cmd([sys.executable, "-m", "deepspeed.env_report"], timeout=30)
        if rc == 0:
            # Extract torch/cuda/deepspeed compatibility line
            compat_lines = [l for l in out.split("\n") if "torch" in l.lower() or "cuda" in l.lower()]
            detail = "; ".join(compat_lines[:3]) if compat_lines else "report generated"
            _check("deepspeed env_report", True, detail[:200], severity="info")
        else:
            _check("deepspeed env_report", False, f"exit {rc}: {out[:120]}", severity="warning")
    except ImportError as exc:
        _check("deepspeed import", False, str(exc))


# ── 5. Python Package Import Checks ──────────────────────────────────────────

def check_imports() -> None:
    _header("Python Package Imports")

    required = [
        ("pytorch_lightning", "pytorch_lightning"),
        ("torchmetrics", "torchmetrics"),
        ("scipy", "scipy"),
        ("requests", "requests"),
        ("click", "click"),
        ("pydantic", "pydantic"),
        ("pyyaml", "yaml"),
        ("numpy", "numpy"),
        ("biopython", "Bio"),
        ("h5py", "h5py"),
        ("pandas", "pandas"),
        ("rich", "rich"),
        ("tqdm", "tqdm"),
    ]

    for label, modname in required:
        try:
            mod = importlib.import_module(modname)
            ver = getattr(mod, "__version__", "ok")
            _check(f"import {label}", True, f"v{ver}")
        except ImportError as exc:
            _check(f"import {label}", False, str(exc))


# ── 6. SSL Probe (report-only, no bypass) ────────────────────────────────────

def check_ssl_probe() -> None:
    _header("SSL Connectivity (report-only)")

    host = "api.colabfold.com"
    port = 443

    # Check system CA bundle
    import certifi
    ca_path = certifi.where()
    _check("certifi CA bundle", Path(ca_path).is_file(), ca_path, severity="info")

    # Attempt SSL handshake
    ctx = ssl.create_default_context()  # uses system CAs; NO verify bypass
    try:
        with socket.create_connection((host, port), timeout=10) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                cert = ssock.getpeercert()
                subject = dict(x[0] for x in cert.get("subject", []))
                issuer = dict(x[0] for x in cert.get("issuerName", cert.get("issuer", [])))
                cn = subject.get("commonName", "unknown")
                _check(f"SSL handshake to {host}", True, f"CN={cn}")
    except ssl.SSLCertVerificationError as exc:
        _check(f"SSL handshake to {host}", False,
               f"Certificate verification failed: {exc}", severity="warning")
    except (socket.timeout, socket.gaierror, OSError) as exc:
        _check(f"SSL handshake to {host}", False,
               f"Connection failed (network issue, not SSL): {exc}", severity="warning")

    # Attempt HTTPS GET via requests (no verify override)
    try:
        import requests
        resp = requests.get(f"https://{host}/", timeout=10)
        _check(f"HTTPS GET {host}", True, f"status={resp.status_code}", severity="info")
    except Exception as exc:
        _check(f"HTTPS GET {host}", False, str(exc)[:120], severity="warning")


# ── 7. File / Path Existence (tolerant of missing) ───────────────────────────

def check_paths() -> None:
    _header("File / Path Existence (tolerant)")

    repo_root = Path(__file__).resolve().parent.parent

    expected = [
        ("configs/smoke_test.yaml", "error"),
        ("pipeline/__init__.py", "error"),
        ("pipeline/orchestrator.py", "error"),
        ("pipeline/schemas.py", "error"),
        ("pipeline/adapters/__init__.py", "error"),
        ("pipeline/adapters/openfold3.py", "error"),
        ("pipeline/adapters/affinity.py", "error"),
        ("scripts/run_pipeline.py", "error"),
        ("environment.yml", "warning"),
        ("requirements-lock.txt", "warning"),
        ("Makefile", "warning"),
        ("runs/", "info"),
    ]

    for rel_path, severity in expected:
        full = repo_root / rel_path
        exists = full.exists()
        _check(f"Path: {rel_path}", exists,
               f"{'found' if exists else 'MISSING'} -> {full}",
               severity=severity)

    # Check for known log paths (tolerant — these may not exist)
    known_log_dirs = [
        Path("/tmp/of3test/out_clean6/logs"),
        repo_root / "runs",
    ]
    for log_dir in known_log_dirs:
        if log_dir.is_dir():
            logs = list(log_dir.glob("*.log")) + list(log_dir.glob("**/predict_err_rank*.log"))
            _check(f"Log dir: {log_dir}", True,
                   f"{len(logs)} log file(s) found", severity="info")
        else:
            _check(f"Log dir: {log_dir}", False,
                   "directory does not exist (this is OK for fresh installs)", severity="info")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="OpenFold3+DeepSpeed environment preflight checks")
    parser.add_argument("--json", action="store_true", help="Print JSON summary to stdout")
    parser.add_argument("--output", type=str, default=None, help="Write JSON report to file")
    args = parser.parse_args()

    print("OpenFold3 + DeepSpeed — Preflight Environment Check")
    print(f"Timestamp: {__import__('datetime').datetime.now().isoformat()}")

    check_python_os()
    check_gpu_cuda()
    check_cuda_headers_libs()
    check_torch_deepspeed()
    check_imports()
    check_ssl_probe()
    check_paths()

    # Summary
    summary = report.summary()
    _header("SUMMARY")
    print(f"  Total: {summary['total_checks']}  "
          f"Passed: {summary['passed']}  "
          f"Failed: {summary['failed']}  "
          f"Warnings: {summary['warnings']}")
    print(f"  Overall: {summary['overall']}")

    if args.json:
        print("\n" + json.dumps(summary, indent=2))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n  Report written to: {out_path}")

    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    main()
