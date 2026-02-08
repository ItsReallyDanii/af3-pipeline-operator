#!/usr/bin/env python3
"""orchestrator_dry.py — Dry-run validation for the AF3 pipeline.

Validates config, schemas, required inputs, and output folder structure.
Writes manifest.json with status=DRY_RUN_OK or DRY_RUN_FAIL.

NEVER imports or runs heavy model code paths (no torch, no deepspeed,
no openfold3, no affinity scoring).

Exit codes:
    0  DRY_RUN_OK   — all validations passed
    1  DRY_RUN_FAIL — one or more validations failed

Usage:
    python orchestrator_dry.py --config configs/smoke_test.yaml --output-base runs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

# Only import lightweight dependencies (no torch, no deepspeed)
try:
    import yaml
except ImportError:
    print("FATAL: pyyaml not installed. Run: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

try:
    from pydantic import BaseModel, Field, ValidationError
    from typing import Literal
except ImportError:
    print("FATAL: pydantic not installed. Run: pip install pydantic", file=sys.stderr)
    sys.exit(1)


# ── Schema (duplicated from pipeline.schemas to avoid importing pipeline) ─────

class InputBlock(BaseModel):
    protein_sequence: str = Field(min_length=20)
    ligand_smiles: str = Field(min_length=1)


class SettingsBlock(BaseModel):
    cuda_device: int = 0
    output_format: Literal["pdb", "cif"] = "pdb"
    seed: int = 42


class RunConfig(BaseModel):
    run_id: str
    mode: Literal["mock", "production"] = "mock"
    target_name: str
    input: InputBlock
    settings: SettingsBlock = SettingsBlock()


# ── Validation checks ────────────────────────────────────────────────────────

@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class DryRunResult:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append(Check(name=name, passed=passed, detail=detail))

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def validate_config_file(config_path: Path, result: DryRunResult) -> dict | None:
    """Validate that the config file exists, is valid YAML, and passes schema."""
    # File existence
    if not config_path.is_file():
        result.add("config_file_exists", False, f"not found: {config_path}")
        return None
    result.add("config_file_exists", True, str(config_path))

    # YAML parse
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        result.add("config_yaml_valid", False, f"YAML parse error: {exc}")
        return None
    result.add("config_yaml_valid", True, "parsed successfully")

    if not isinstance(raw, dict):
        result.add("config_is_dict", False, f"expected dict, got {type(raw).__name__}")
        return None
    result.add("config_is_dict", True, f"{len(raw)} top-level keys")

    # Schema validation
    try:
        validated = RunConfig(**raw)
        result.add("config_schema_valid", True,
                    f"run_id={validated.run_id}, mode={validated.mode}")
    except ValidationError as exc:
        result.add("config_schema_valid", False, str(exc))
        return raw  # Return raw so we can still do partial checks

    return raw


def validate_required_paths(repo_root: Path, result: DryRunResult) -> None:
    """Check that essential project files exist."""
    required = [
        "pipeline/__init__.py",
        "pipeline/orchestrator.py",
        "pipeline/schemas.py",
        "pipeline/adapters/__init__.py",
        "pipeline/adapters/openfold3.py",
        "pipeline/adapters/affinity.py",
        "scripts/run_pipeline.py",
    ]
    for rel in required:
        full = repo_root / rel
        exists = full.is_file()
        result.add(f"path:{rel}", exists,
                    "found" if exists else f"MISSING: {full}")


def validate_output_base(output_base: Path, result: DryRunResult) -> None:
    """Validate the output base directory is writable."""
    if output_base.exists():
        if output_base.is_dir():
            result.add("output_base_exists", True, str(output_base))
        else:
            result.add("output_base_exists", False,
                        f"exists but is not a directory: {output_base}")
            return
    else:
        result.add("output_base_exists", True,
                    f"will be created: {output_base}")

    # Test writability
    try:
        output_base.mkdir(parents=True, exist_ok=True)
        test_file = output_base / ".dry_run_write_test"
        test_file.write_text("test", encoding="utf-8")
        test_file.unlink()
        result.add("output_base_writable", True, str(output_base))
    except OSError as exc:
        result.add("output_base_writable", False, str(exc))


def validate_input_data(config: dict, result: DryRunResult) -> None:
    """Validate the input data fields from config."""
    inp = config.get("input", {})

    seq = inp.get("protein_sequence", "")
    if len(seq) >= 20:
        result.add("input_sequence_length", True, f"{len(seq)} residues")
    else:
        result.add("input_sequence_length", False,
                    f"{len(seq)} residues (minimum 20)")

    # Basic amino acid alphabet check (single-letter codes)
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    invalid_chars = set(seq.upper()) - valid_aa
    if not invalid_chars:
        result.add("input_sequence_alphabet", True, "all standard amino acids")
    else:
        result.add("input_sequence_alphabet", False,
                    f"non-standard characters: {invalid_chars}")

    smiles = inp.get("ligand_smiles", "")
    if len(smiles) >= 1:
        result.add("input_smiles_present", True, f"{len(smiles)} chars")
    else:
        result.add("input_smiles_present", False, "empty SMILES string")


def create_output_structure(output_base: Path, run_id: str,
                            result: DryRunResult) -> Path:
    """Create the output folder structure for the run."""
    run_dir = output_base / run_id
    logs_dir = run_dir / "logs"

    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        result.add("output_dirs_created", True, str(run_dir))
    except OSError as exc:
        result.add("output_dirs_created", False, str(exc))

    return run_dir


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dry-run validation for AF3 pipeline (no inference)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to run config YAML")
    parser.add_argument("--output-base", type=str, default="runs",
                        help="Base directory for run outputs")
    args = parser.parse_args()

    start_time = time.time()
    repo_root = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    output_base = Path(args.output_base)
    if not output_base.is_absolute():
        output_base = repo_root / output_base

    result = DryRunResult()

    print("=" * 60)
    print("  OpenFold3 + DeepSpeed — Dry-Run Validation")
    print("  (No model loading, no inference, no GPU usage)")
    print("=" * 60)
    print()

    # 1. Config validation
    print("[1/4] Validating config file...")
    config = validate_config_file(config_path, result)

    # 2. Required paths
    print("[2/4] Checking required project paths...")
    validate_required_paths(repo_root, result)

    # 3. Output directory
    print("[3/4] Validating output directory...")
    validate_output_base(output_base, result)

    # 4. Input data
    if config:
        print("[4/4] Validating input data...")
        validate_input_data(config, result)
    else:
        result.add("input_data_validation", False, "skipped — config invalid")

    # Create output structure
    run_id = config.get("run_id", "dry_run") if config else "dry_run"
    run_dir = create_output_structure(output_base, run_id, result)

    # Build manifest
    status = "DRY_RUN_OK" if result.all_passed else "DRY_RUN_FAIL"
    duration = round(time.time() - start_time, 3)

    manifest = {
        "run_id": run_id,
        "status": status,
        "mode": "dry_run",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": duration,
        "config_path": str(config_path),
        "config_hash_sha256": _hash_file(config_path) if config_path.is_file() else "N/A",
        "output_dir": str(run_dir),
        "checks": [asdict(c) for c in result.checks],
        "summary": {
            "total": len(result.checks),
            "passed": sum(1 for c in result.checks if c.passed),
            "failed": sum(1 for c in result.checks if not c.passed),
        },
        "reproducibility": {
            "pipeline_git_sha": _git_sha(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
    }

    # Write manifest
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Print results
    print()
    print("-" * 60)
    for c in result.checks:
        icon = "PASS" if c.passed else "FAIL"
        print(f"  [{icon}] {c.name}: {c.detail}")
    print("-" * 60)
    print(f"  Status:   {status}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Duration: {duration}s")
    print()

    sys.exit(0 if result.all_passed else 1)


if __name__ == "__main__":
    main()
