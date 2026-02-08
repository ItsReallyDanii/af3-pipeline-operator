from pathlib import Path

def run(structure_pdb: Path, smiles: str, mode: str = "mock", settings: dict | None = None) -> dict:
    if mode == "mock":
        return {"pKd": 8.5, "confidence": 0.90, "source": "mock"}

    # Production stub: wire to AQAffinity scoring call
    raise NotImplementedError("Production mode not wired yet. Implement AQAffinity invocation.")
