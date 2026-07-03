# Running the PolyMath eval on the DIAS cluster (Slurm)

Cluster docs: https://uclphysast.github.io/clusters/dias/

- Scheduler: **Slurm**. Never run generation on the login node — submit with `sbatch`.
- GPU partitions: **GPU** (3x A100 40GB) or **LIGHTGPU** (same cards sliced into 6x 20GB,
  request 1 GPU per job). A 1.5B model comfortably fits a 20GB slice, so start with LIGHTGPU
  for shorter queue times; fall back to GPU if you hit OOM.
- Storage on DIAS is **not backed up**. Since you're only here temporarily, copy
  `logs-eval/` and `Plots/` off the cluster before you switch to your next HPC (see step 4).

## 1. One-time setup (on the login node)

```bash
ssh <user>@dias.hpc.phys.ucl.ac.uk

git clone https://github.com/Rauljo/MuLE.git
cd MuLE
git checkout polymath_evaluation
```

DIAS has no `conda`/Miniforge and only a system Python 3.9 module (this repo needs 3.11), but
`apptainer` is available at `/usr/bin/apptainer` — use it to get a clean Python 3.11 runtime
without needing admin rights or a package that matches the module system. This also makes the
whole setup portable to your next HPC: the same two commands below reproduce it anywhere
Apptainer is installed, with no cluster-specific package manager involved.

Pull a small base image (not the 7-12GB `vllm/vllm-openai` image — its published tags don't
match this repo's pinned `vllm` version anyway; `pip` resolves the correct one from PyPI, and
vLLM's wheels bundle their own CUDA runtime, so all the container needs to provide is Python
3.11 + the host's NVIDIA driver via `--nv`):

```bash
mkdir -p "$HOME/containers"
apptainer pull "$HOME/containers/python311.sif" docker://python:3.11-slim
```

Create a venv on the host filesystem using the container's Python, then install into it
(the venv lives in `$HOME`, outside the read-only `.sif`, so it persists across jobs):

```bash
apptainer exec "$HOME/containers/python311.sif" python3 -m venv "$HOME/mule_venv"

apptainer exec "$HOME/containers/python311.sif" bash -c "
  source '$HOME/mule_venv/bin/activate'
  pip install --upgrade pip
  pip install -r evaluation/requirements-py311.txt
  pip install fasttext-wheel peft pandas seaborn matplotlib
"

mkdir -p evaluation/eval_tools/langid
curl -L https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz \
  -o evaluation/eval_tools/langid/lid.176.ftz
```

Pre-download the model on the login node, so the GPU job doesn't depend on the compute
node having outbound internet:

```bash
apptainer exec "$HOME/containers/python311.sif" bash -c "
  source '$HOME/mule_venv/bin/activate'
  python - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download('XueZhang-bjtu/1.5B-cold-start-SFT')
print(path)
PY
"
```

Note the printed path — use it as `MODEL_PATH` below (or keep the HF repo id directly if you
confirm compute nodes do have internet access).

Every subsequent Python invocation in this doc runs the same way:
`apptainer exec --nv "$HOME/containers/python311.sif" bash -c "source '$HOME/mule_venv/bin/activate' && <command>"`
— `--nv` is only needed (and only works) on a GPU node, so it's omitted above for the
login-node setup steps.

## 2. Slurm job: generation + scoring

The actual pipeline (generation, scoring, per-level report — scoped to `en es fr pt`) lives in
[run_polymath_pipeline.sh](run_polymath_pipeline.sh), a plain bash script with no
container/Slurm specifics, so it's easy to read/test/reuse on its own. It does **not** use
`scripts-test/gen_PolyMath_res.sh` / `eval_tools/PolyMath/eval/run_eval.sh`, since those
hardcode the full 10-language benchmark and would waste GPU time on languages you don't need.

Save this as `evaluation/run_dias_polymath.sbatch` — it just wraps that script in the
Apptainer environment and points it at a GPU:

```bash
#!/bin/bash -l
#SBATCH -p LIGHTGPU
#SBATCH -N1
#SBATCH -n4
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH -o polymath_%j.out
#SBATCH -e polymath_%j.err

source /etc/slurm/gpu_variables.sh 2>/dev/null || true
cd "$SLURM_SUBMIT_DIR"

SIF="$HOME/containers/python311.sif"
VENV="$HOME/mule_venv"
MODEL_PATH="XueZhang-bjtu/1.5B-cold-start-SFT"   # or the local snapshot path from setup step 1

apptainer exec --nv "$SIF" bash -c "
  source '$VENV/bin/activate'
  MODEL_PATH='$MODEL_PATH' bash run_polymath_pipeline.sh
"
```

Submit and monitor from the `evaluation/` directory:

```bash
cd evaluation
sbatch run_dias_polymath.sbatch
squeue -u $USER
tail -f polymath_<jobid>.out
```

If the default 0.85 GPU memory utilization OOMs on a 20GB LIGHTGPU slice, `run_polymath_pipeline.sh`
reads its memory/batch settings from env vars, so you can override them in the sbatch script
without editing the pipeline itself — e.g. the repo's README documents this conservative
fallback:

```bash
apptainer exec --nv "$SIF" bash -c "
  source '$VENV/bin/activate'
  MODEL_PATH='$MODEL_PATH' GPU_MEM_UTIL=0.35 MAX_MODEL_LEN=2048 MAX_TOKENS=512 \
    MAX_NUM_SEQS=1 MAX_NUM_BATCHED_TOKENS=512 \
    EXTRA_GEN_ARGS='--enforce_eager --disable_prefix_caching' \
    bash run_polymath_pipeline.sh
"
```

Or resubmit with `-p GPU` for a full 40GB card instead.

## 3. Backtracking/token/LC/accuracy analysis

CPU-only, can run directly on the login node once the job above finishes (no `--nv` needed):

```bash
cd "PolyMath Answers Statistics"
apptainer exec "$HOME/containers/python311.sif" bash -c "
  source '$HOME/mule_venv/bin/activate'
  python3 Total_answers_stats.py
  python3 Per_question_stats.py
"
```

## 4. Before you move to your next HPC

DIAS storage is not backed up. Copy off, e.g. from your local machine:

```bash
scp -r <user>@dias.hpc.phys.ucl.ac.uk:~/MuLE/evaluation/logs-eval ./
scp -r "<user>@dias.hpc.phys.ucl.ac.uk:~/MuLE/PolyMath Answers Statistics/Plots" ./
```

Any code changes you make on DIAS should be committed and pushed to the `polymath_evaluation`
branch on GitHub — that's your durable copy, independent of which cluster you're on next.

The Apptainer setup itself is also portable: on your next HPC (as long as Apptainer/Singularity
is installed there), just re-run the two setup commands from step 1 — `apptainer pull` +
`python3 -m venv` + `pip install` — rather than copying `$HOME/containers/python311.sif` and
`$HOME/mule_venv` over. `pip install` builds an environment from scratch quickly since it's
just a small base image and pure pip installs; the venv reinstalling is more reliable than
transferring it (a venv's paths are baked in at creation, so a copied one only works if the new
machine's paths line up exactly).