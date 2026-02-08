# AF3 Pipeline Operator (Scaffold v1)

This repository is a reproducible scaffold for a structure→affinity pipeline:

- Input: protein sequence + ligand SMILES
- Step 1: OpenFold-3 inference adapter
- Step 2: Affinity scoring adapter
- Output: `runs/<run_id>/manifest.json`

## Project Structure

```
af3-pipeline-operator/
├── configs/
├── docker/
├── pipeline/
│   ├── adapters/
│   └── orchestrator.py
├── scripts/
├── requirements.txt
└── README.md
```

## Local Smoke Test (Mock Mode)

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py --config configs/smoke_test.yaml --output ./runs
```

Success criteria:
- `runs/smoke_test_001/manifest.json` exists
- `status` is `SUCCESS`

## Docker Build (Camber)

Use repo root as build context:

```bash
docker build -t af3-op -f docker/Dockerfile .
```

Run with mount:

```bash
docker run --gpus all \
  -v $(pwd):/app/pipeline \
  -w /app/pipeline \
  af3-op \
  python scripts/run_pipeline.py --config configs/smoke_test.yaml --output ./runs
```

## Notes

- `mode: mock` is for local orchestration only.
- `mode: production` requires implementing adapters for real OpenFold-3 and AQAffinity calls.
- OpenFold-3 commit is pinned in Dockerfile (`aeac5fd`) for determinism.
