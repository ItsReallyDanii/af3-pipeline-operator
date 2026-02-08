from pydantic import BaseModel, Field
from typing import Literal

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
