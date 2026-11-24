# CUDA Docker runtime

The benchmark image is pinned to CUDA 12.4, Python 3.11, and a CUDA-enabled
PyTorch build. `uv` installs the interpreter and creates `/opt/venv`; all
project packages are installed with `/opt/venv/bin/pip3`. Conda and the Mamba
environment manager are not used (`mamba-ssm` is the model library).
The image tag is `fin-dl-benchmark:cuda12.4-india1`; an older image tag does
not satisfy the runner's image check.

## Host requirements

- NVIDIA driver compatible with CUDA 12.4.
- Docker Desktop/Engine with Compose v2 and NVIDIA Container Toolkit support.
- Access to the selected dataset; the runners prepare missing Indian or FRED
  inputs under `data/`. See [RunningNotebooks.md](RunningNotebooks.md) for
  source-specific files and optional preparation commands.

## Run

From Git Bash or WSL at the repository root, set a private, nonempty token in
the same shell before running any Compose or runner command. Compose requires
`JUPYTER_TOKEN` even when the benchmark is launched without opening JupyterLab:

```bash
read -rs -p 'Jupyter token: ' JUPYTER_TOKEN
export JUPYTER_TOKEN
```

```bash
DATASET_MODE=india_indices FOLDS= ./run_all_models.sh
```

`india_indices` is the default dataset. The missing-value and cross-market
alignment guards described in [RunningNotebooks.md](RunningNotebooks.md)
apply to the separate legacy `global_indian` selection.

When invoked on the host, the script checks the Docker daemon, reuses the
local image or builds it if missing, starts the Jupyter
service, and re-enters itself inside the container. It then runs CUDA/Mamba
preflight, data audit, one benchmark notebook per model/seed/fold, validation,
report generation, checksums, and ZIP integrity checks.

Common overrides:

```bash
DATASET_MODE=india_indices RUN_ID=pilot ESTIMATE_ONLY=1 ./run_all_models.sh
DATASET_MODE=india_indices RUN_ID=smoke MODELS=dlinear,lstm SEEDS=42 FOLDS=0 EPOCHS=2 ./run_all_models.sh
DATASET_MODE=india_indices RUN_ID=full DEVICE=gpu RESUME_MODE=auto MAX_RETRIES=2 ./run_all_models.sh
```

`DEVICE=gpu` fails fast when CUDA is unavailable. CPU use must be explicit with
`DEVICE=cpu`; this selects CPU computation inside the container. The supplied
Compose service still requests `gpus: all`, so its NVIDIA host requirements
also apply to CPU runs. `ESTIMATE_ONLY=1` measures one epoch per selected model
using the first selected seed and fold on the selected device.
`RESUME_MODE` is `auto`, `restart`, or `fail`. Raw data is omitted
from the final ZIP; `INCLUDE_DATA=1` is rejected. Checkpoints are included
by default; `PACKAGE_CHECKPOINTS=0` excludes them.

JupyterLab is available separately with:

```bash
docker compose up --build jupyter
```

Open <http://localhost:8888> and select **Financial DL Benchmark (CUDA)**.
Clean templates are under `notebooks/`; executed copies are always stored under
the selected run directory.

## Output contract

Successful full runs produce `outputs/<run_id>/`, root `Final_Metrics.md`, and
`artifacts/<run_id>.zip`. A timing-only pilot reports `pilot_complete` and
does not create a ZIP. Failed runs preserve partial telemetry and
checkpoints, return a nonzero status, and never create a ZIP.
