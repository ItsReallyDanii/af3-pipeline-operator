import yaml
import json
import logging
import time
import hashlib
import os
import platform
import subprocess
from pathlib import Path
from datetime import datetime
from pipeline.adapters import openfold3, affinity

logging.basicConfig(level=logging.INFO, format="[OPERATOR] %(message)s")
logger = logging.getLogger(__name__)

class PipelineOperator:
    def __init__(self, config_path: str, output_base: str):
        self.start_time = time.time()
        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)
        self.run_id = self.config.get("run_id", f"run_{int(self.start_time)}")
        self.output_dir = Path(output_base) / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.manifest = {
            "run_id": self.run_id,
            "timestamp_start_utc": datetime.utcnow().isoformat() + "Z",
            "config_hash_sha256": self._hash_file(self.config_path),
            "environment_mode": self.config.get("mode", "mock"),
            "status": "INIT",
            "steps": {},
            "errors": [],
            "reproducibility": self._reproducibility_block()
        }

    def _load_config(self, path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _hash_file(self, path: Path) -> str:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def _git_sha(self) -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            return "unknown"

    def _reproducibility_block(self) -> dict:
        return {
            "pipeline_git_sha": self._git_sha(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "openfold3_commit": "aeac5fd",
            "container_image": os.environ.get("AF3_IMAGE_TAG", "unknown"),
        }

    def execute(self) -> None:
        try:
            logger.info("STEP 1/2: OpenFold-3 inference")
            structure_file = openfold3.run(
                sequence=self.config["input"]["protein_sequence"],
                output_dir=self.output_dir,
                mode=self.config.get("mode", "mock"),
                settings=self.config.get("settings", {})
            )
            self.manifest["steps"]["openfold3"] = {
                "status": "SUCCESS",
                "output_structure": str(structure_file)
            }

            logger.info("STEP 2/2: Affinity scoring")
            score_data = affinity.run(
                structure_pdb=structure_file,
                smiles=self.config["input"]["ligand_smiles"],
                mode=self.config.get("mode", "mock"),
                settings=self.config.get("settings", {})
            )
            self.manifest["steps"]["affinity"] = {
                "status": "SUCCESS",
                "scores": score_data
            }

            self.manifest["status"] = "SUCCESS"

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.manifest["status"] = "FAILED"
            self.manifest["errors"].append(str(e))
            raise
        finally:
            self._save_manifest()

    def _save_manifest(self) -> None:
        self.manifest["timestamp_end_utc"] = datetime.utcnow().isoformat() + "Z"
        self.manifest["duration_seconds"] = round(time.time() - self.start_time, 2)

        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2)
        logger.info(f"Manifest saved: {manifest_path}")
