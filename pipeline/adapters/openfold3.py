from pathlib import Path
import time

def run(sequence: str, output_dir: Path, mode: str = "mock", settings: dict | None = None) -> Path:
    output_path = Path(output_dir) / "prediction.pdb"

    if mode == "mock":
        # Minimal valid-ish placeholder file
        output_path.write_text("HEADER    MOCK PDB\nEND\n", encoding="utf-8")
        time.sleep(0.2)
        return output_path

    # Production stub: wire to OpenFold-3 CLI/API on Camber
    raise NotImplementedError("Production mode not wired yet. Implement OpenFold-3 invocation.")
