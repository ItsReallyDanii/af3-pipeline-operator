# Troubleshooting Guide — OpenFold3 + DeepSpeed Environment

## Table of Contents

1. [Missing Log File Handling](#1-missing-log-file-handling)
2. [CUBLAS_STATUS_INVALID_VALUE Decision Tree](#2-cublas_status_invalid_value-decision-tree)
3. [Missing CUDA Headers Decision Tree](#3-missing-cuda-headers-cusolverdnh--cusparseh)
4. [Dependency Conflict Matrix](#4-dependency-conflict-matrix)
5. [SSL Certificate Failure Policy](#5-ssl-certificate-failure-policy)

---

## 1. Missing Log File Handling

### Symptom

A referenced log file does not exist, e.g.:

```
FileNotFoundError: /tmp/of3test/out_clean6/logs/predict_err_rank0.log
```

### Why This Happens

- The run never started (config error, OOM before first write, CUDA init failure).
- The run started but crashed before log rotation flushed to disk.
- The output directory was cleaned or never created.
- The path is from a different machine / container / prior experiment.

### Decision Tree

```
Log file missing?
├── Does the parent directory exist?
│   ├── NO  → The run never created its output tree.
│   │         Check: was orchestrator_dry.py / run_pipeline.py invoked?
│   │         Action: run `make run.dry` to verify config creates dirs.
│   │
│   └── YES → Directory exists but log is absent.
│       ├── Is manifest.json present in the run dir?
│       │   ├── YES → Read manifest.json "status" field.
│       │   │   ├── status=SUCCESS → Run completed; logs may have been
│       │   │   │                     written to stdout only. Check:
│       │   │   │                     `journalctl`, `dmesg`, or container logs.
│       │   │   └── status=FAILED  → Check manifest "errors" array.
│       │   │                        The error may have occurred before
│       │   │                        log file creation.
│       │   └── NO  → Run crashed before manifest was written.
│       │             Check: `dmesg | grep -i oom`
│       │             Check: `nvidia-smi` for Xid errors
│       │             Check: `journalctl -u <service> --since "1 hour ago"`
│       │
│       └── Are there ANY files in the run dir?
│           ├── NO  → mkdir succeeded but nothing else ran.
│           │         Likely a very early crash (import error, CUDA init).
│           │         Action: run `python scripts/preflight.py` to diagnose.
│           └── YES → Partial run. Inspect whatever files exist.
│                     Look for: *.pdb, *.cif, stderr captures, core dumps.
```

### What to Inspect Next

1. **`make diag.collect`** — generates a full diagnostics bundle.
2. **`dmesg | tail -50`** — kernel OOM killer or GPU errors.
3. **`nvidia-smi -q -d ECC,PAGE_RETIREMENT`** — hardware errors.
4. **Container/systemd logs** — if running in Docker or as a service.
5. **`ls -laR <output_dir>/`** — see what partial artifacts exist.

---

## 2. CUBLAS_STATUS_INVALID_VALUE Decision Tree

### Symptom

```
RuntimeError: CUBLAS_STATUS_INVALID_VALUE when calling cublasSgemm
```

or similar cuBLAS errors during forward pass or DeepSpeed operations.

### Decision Tree

```
CUBLAS_STATUS_INVALID_VALUE
│
├── 1. Check tensor shapes
│   │   Action: Add shape logging before the failing op.
│   │   Common cause: batch dim = 0, mismatched matrix dims,
│   │   or integer overflow in very large tensors.
│   └── Shapes look correct?
│       │
│       ├── 2. Check CUDA memory
│       │   Action: `nvidia-smi` and `torch.cuda.memory_summary()`
│       │   Common cause: fragmented VRAM, near-OOM condition.
│       │   Fix: reduce batch size, enable gradient checkpointing,
│       │        or set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
│       │
│       ├── 3. Check dtype mismatches
│       │   Action: Verify both operands are the same dtype.
│       │   Common cause: mixing fp16 and fp32 without autocast.
│       │   Fix: wrap in `torch.autocast('cuda')` or cast explicitly.
│       │
│       ├── 4. Check cuBLAS version compatibility
│       │   Action: `python -c "import torch; print(torch.version.cuda)"`
│       │   Compare against: `nvcc --version` and `nvidia-smi` driver CUDA.
│       │   Rule: PyTorch CUDA version <= driver CUDA version.
│       │   Fix: if mismatched, rebuild env with correct CUDA toolkit.
│       │
│       ├── 5. Check cuBLAS workspace
│       │   Action: `export CUBLAS_WORKSPACE_CONFIG=:16:8`
│       │   This limits workspace to prevent allocation failures.
│       │
│       └── 6. Check for NaN/Inf inputs
│           Action: `torch.autograd.set_detect_anomaly(True)`
│           Common cause: NaN propagated from earlier layer.
│           Fix: find and fix the source of NaN (loss scaling, LR).
```

### Quick Fixes to Try (in order)

```bash
# 1. Set deterministic cuBLAS workspace
export CUBLAS_WORKSPACE_CONFIG=:16:8

# 2. Enable expandable memory segments
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 3. Verify CUDA version compatibility
python -c "
import torch
print(f'PyTorch CUDA: {torch.version.cuda}')
print(f'cuDNN: {torch.backends.cudnn.version()}')
print(f'Device: {torch.cuda.get_device_name(0)}')
props = torch.cuda.get_device_properties(0)
print(f'Compute capability: {props.major}.{props.minor}')
"

# 4. Run preflight to check full stack
python scripts/preflight.py
```

---

## 3. Missing CUDA Headers (cusolverDn.h / cusparse.h)

### Symptom

```
fatal error: cusolverDn.h: No such file or directory
```

or

```
fatal error: cusparse.h: No such file or directory
```

These occur when compiling CUDA extensions (DeepSpeed ops, custom kernels).

### Decision Tree

```
Missing CUDA header?
│
├── 1. Is CUDA_HOME set correctly?
│   Action: echo $CUDA_HOME
│   Expected: /usr/local/cuda-12.6 or $CONDA_PREFIX
│   ├── Not set → export CUDA_HOME=$CONDA_PREFIX
│   └── Set but wrong path → fix to match actual installation
│
├── 2. Are dev packages installed?
│   Action (conda env):
│     conda list | grep -E 'cuda-libraries-dev|libcusolver-dev|libcusparse-dev'
│   Expected: all three present.
│   ├── Missing → conda install -n of3ds cuda-libraries-dev libcusolver-dev libcusparse-dev
│   └── Present → check include paths (step 3)
│
├── 3. Do the headers exist on disk?
│   Action:
│     find $CONDA_PREFIX -name "cusolverDn.h" 2>/dev/null
│     find $CONDA_PREFIX -name "cusparse.h" 2>/dev/null
│   ├── Found in non-standard path →
│   │   export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH
│   │   export LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:$LIBRARY_PATH
│   └── Not found anywhere → dev packages are broken; reinstall:
│       conda install --force-reinstall -n of3ds cuda-libraries-dev
│
├── 4. Is the compiler finding them?
│   Action: echo | gcc -xc -E -v - 2>&1 | grep include
│   Check that $CONDA_PREFIX/include is in the search path.
│   Fix: export C_INCLUDE_PATH=$CONDA_PREFIX/include
│        export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include
│
└── 5. DeepSpeed-specific: are ops being JIT compiled?
    Action: ds_report | grep "compatible"
    Fix: pre-build DeepSpeed ops:
      DS_BUILD_OPS=1 pip install --no-cache-dir deepspeed==0.16.2
    Or set: export DS_BUILD_UTILS=1 DS_BUILD_FUSED_ADAM=1
```

### Environment Variables Checklist

```bash
# Verify these are set in your conda env activation:
export CUDA_HOME="${CONDA_PREFIX}"
export CPATH="${CONDA_PREFIX}/targets/x86_64-linux/include:${CPATH:-}"
export LIBRARY_PATH="${CONDA_PREFIX}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
```

---

## 4. Dependency Conflict Matrix

Known conflicts between packages in the OpenFold3 + DeepSpeed stack.

### numpy / scipy / numba

| Package | Requires numpy | Notes |
|---------|---------------|-------|
| scipy==1.14.1 | >=1.23.5, <2.1 | Works with numpy 1.26.x |
| numba==0.61.0 | >=1.24, <1.28 | **Breaks with numpy 2.x** |
| torch==2.6.0 | >=1.22.0 | Works with numpy 1.26.x or 2.x |
| pandas==2.2.3 | >=1.23.2 | Works with numpy 1.26.x or 2.x |
| scikit-learn==1.6.1 | >=1.21.0 | Works with numpy 1.26.x or 2.x |
| biopython==1.85 | >=1.22.0 | Works with numpy 1.26.x or 2.x |

**Resolution:** Pin `numpy==1.26.4`. This is the latest 1.x release and satisfies
all packages. Do NOT upgrade to numpy 2.x while using numba 0.61.x.

### datasets / fsspec / s3fs / aiobotocore / botocore

| Package | Requires | Notes |
|---------|----------|-------|
| datasets==3.2.0 | fsspec[http]>=2023.1.0,<=2024.12.0 | Upper bound on fsspec |
| s3fs==2024.12.0 | aiobotocore>=2.15.1,<2.15.3 | Very tight pin on aiobotocore |
| aiobotocore==2.15.2 | botocore>=1.35.76,<1.35.77 | Very tight pin on botocore |

**Resolution:** These must be installed together as a compatible set. The pins in
`requirements-lock.txt` are validated. Do NOT independently upgrade any one of
`fsspec`, `s3fs`, `aiobotocore`, or `botocore`.

### Conflict Detection Commands

```bash
# Check for conflicts in current environment
pip check

# Detailed dependency tree
pip install pipdeptree && pipdeptree --warn fail

# Specific package conflict check
pip install --dry-run --no-deps numpy==2.0.0 2>&1 | head -20
```

### If You Hit a Conflict

1. **Do NOT** run `pip install --force-reinstall` on individual packages.
2. **Do** run `make env.diff` to see what drifted from the lockfile.
3. **Do** run `make env.rollback` to restore from the last known-good snapshot.
4. If no snapshot exists, run `make env.rebuild` for a clean slate.

---

## 5. SSL Certificate Failure Policy

### Principle

**We strictly prohibit global SSL verification bypass.** The following patterns
are **NEVER acceptable** in this project:

```python
# FORBIDDEN — do not use anywhere
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# FORBIDDEN — do not use as a default
import requests
requests.get(url, verify=False)

# FORBIDDEN — do not set globally
import os
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
```

### Symptom

```
ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED]
    certificate verify failed: unable to get local issuer certificate
```

### Decision Tree

```
SSL verification fails?
│
├── 1. Is the system CA bundle up to date?
│   Action: python -c "import certifi; print(certifi.where())"
│   Then: ls -la <path>  # check date
│   Fix: pip install --upgrade certifi
│        conda install -n of3ds ca-certificates
│
├── 2. Is there a corporate proxy / MITM?
│   Action: openssl s_client -connect api.colabfold.com:443 -showcerts
│   Look for: corporate CA in the chain
│   Fix: export REQUESTS_CA_BUNDLE=/path/to/corporate-ca-bundle.pem
│        (This is a TARGETED override, not a global bypass)
│
├── 3. Is the clock correct?
│   Action: date && timedatectl status
│   Fix: timedatectl set-ntp true
│
├── 4. Is it a conda vs system Python CA mismatch?
│   Action: python -c "import ssl; print(ssl.get_default_verify_paths())"
│   Fix: export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
│
└── 5. Is the specific endpoint down or misconfigured?
    Action: curl -vI https://api.colabfold.com/ 2>&1 | grep -i ssl
    If the server cert is actually invalid → report upstream, do NOT bypass.
```

### Acceptable SSL Configuration

```python
# ACCEPTABLE — targeted CA bundle for corporate environments
import requests
session = requests.Session()
session.verify = "/etc/pki/tls/certs/corporate-ca-bundle.pem"

# ACCEPTABLE — using certifi explicitly
import certifi
import requests
resp = requests.get(url, verify=certifi.where())

# ACCEPTABLE — environment variable pointing to valid CA bundle
# export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
```

### Preflight SSL Check

The `scripts/preflight.py` tool performs a report-only SSL probe to
`api.colabfold.com`. It will:

- Report the certificate chain details.
- Report success or failure.
- **Never** bypass verification.
- Classify SSL failures as **warnings** (not hard errors) since network
  connectivity may not be available in all environments.

---

## Quick Reference: Environment Recovery Commands

```bash
# Full diagnostics
make diag.collect

# See what's wrong
make env.verify

# See what drifted
make env.diff

# Restore from snapshot
make env.rollback && make env.verify

# Nuclear option: full rebuild
make env.rebuild
```
