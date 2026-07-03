# Running the PolyMath eval on the DIAS cluster (Slurm)

Cluster docs: https://uclphysast.github.io/clusters/dias/

- Scheduler: **Slurm**. Never run generation on the login node — submit with `sbatch`.
- GPU partitions: **GPU** (3x whole A100 80GB cards, on one node: `compute-gpu-0-1`) or
  **LIGHTGPU** (same cards sliced via NVIDIA MIG into 6x 20GB instances). **Use `GPU`, not
  `LIGHTGPU`**: MIG slices are addressed by a UUID string rather than a plain integer device
  index, and this version of vLLM crashes trying to parse that
  (`ValueError: invalid literal for int() ... 'MIG-...'`).
- Storage on DIAS is **not backed up**. Since you're only here temporarily, copy
  `logs-eval/` and `Plots/` off the cluster before you switch to your next HPC (see step 4).

## 1. One-time setup (on the login node)

```bash
ssh <user>@dias.hpc.phys.ucl.ac.uk

git clone https://github.com/Rauljo/MuLE.git
cd MuLE
git checkout polymath_evaluation
```

### Git LFS (required for the PolyMath dataset)

`evaluation/eval_tools/PolyMath/data-polymath/` is tracked with Git LFS. DIAS has no `git-lfs`
installed, so a plain `git clone` leaves you with tiny (~130 byte) pointer-file stubs instead of
the real `.parquet` files — `load_dataset(...)` will fail with
`pyarrow.lib.ArrowInvalid: Parquet magic bytes not found in footer` until you fetch the real
content. Install a portable copy into your own home directory (no admin rights needed):

```bash
cd ~
curl -L -o git-lfs.tar.gz https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-amd64-v3.5.1.tar.gz
tar xzf git-lfs.tar.gz
mkdir -p ~/bin
cp git-lfs-3.5.1/git-lfs ~/bin/
export PATH="$HOME/bin:$PATH"   # add this to ~/.bashrc too, so it's there in future sessions
git lfs version
```

Then fetch the real files:

```bash
cd ~/MuLE
git lfs install --local
git lfs pull
```

Verify: `evaluation/eval_tools/PolyMath/data-polymath/en/low.parquet` should be ~22KB (matching
the `size` field the pointer file used to show), not 130 bytes, and should start with the
magic bytes `PAR1`:

```bash
head -c 4 evaluation/eval_tools/PolyMath/data-polymath/en/low.parquet
```

### Python environment (Apptainer)

DIAS has no `conda`/Miniforge and only a system Python 3.9 module (this repo needs 3.11), but
`apptainer` is available at `/usr/bin/apptainer` — use it to get a clean Python 3.11 runtime
without needing admin rights or a package that matches the module system. This also makes the
whole setup portable to your next HPC: the same commands below reproduce it anywhere Apptainer
is installed, with no cluster-specific package manager involved.

Pull a base image (not the 7-12GB `vllm/vllm-openai` image — its published tags don't match
this repo's pinned `vllm` version anyway; `pip` resolves the correct one from PyPI, and vLLM's
wheels bundle their own CUDA runtime, so the container mainly needs to provide Python 3.11 +
the host's NVIDIA driver via `--nv`). Use the **non-slim** `python:3.11` image, not
`python:3.11-slim`: vLLM's engine uses Triton to JIT-compile some kernels (the top-k/top-p
sampler, and — unless `--enforce_eager` is set — the compiled model graph too), which needs a
C compiler at runtime. `-slim` strips that out to save space; the non-slim image is built on
`buildpack-deps`, which includes `gcc`:

```bash
mkdir -p "$HOME/containers"
apptainer pull "$HOME/containers/python311.sif" docker://python:3.11
```

Create a venv on the host filesystem using the container's Python, then install into it
(the venv lives in `$HOME`, outside the read-only `.sif`, so it persists across jobs):

```bash
apptainer exec "$HOME/containers/python311.sif" python3 -m venv "$HOME/mule_venv"

apptainer exec "$HOME/containers/python311.sif" bash -c "
  source '$HOME/mule_venv/bin/activate'
  pip install --upgrade pip
  pip install 'vllm==0.16.0'
  pip install -r evaluation/requirements-py311.txt
  pip install fasttext-wheel peft pandas seaborn matplotlib
"

mkdir -p evaluation/eval_tools/langid
curl -L https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz \
  -o evaluation/eval_tools/langid/lid.176.ftz
```

`vllm==0.16.0` is pinned explicitly (installed first, before `requirements-py311.txt`, so it
isn't upgraded to a newer version by that file's looser `>=0.16.0,<0.18` range) — this is the
specific version that's been confirmed working end-to-end on this cluster. See the
troubleshooting notes at the bottom for why this matters.

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
#SBATCH -p GPU
#SBATCH -N1
#SBATCH -n4
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH -o polymath_%j.out
#SBATCH -e polymath_%j.err

source /etc/slurm/gpu_variables.sh 2>/dev/null || true
cd "$SLURM_SUBMIT_DIR"

SIF="$HOME/containers/python311.sif"
VENV="$HOME/mule_venv"
MODEL_PATH="XueZhang-bjtu/1.5B-cold-start-SFT"   # or the local snapshot path from setup step 1

ulimit -l unlimited

apptainer exec --nv "$SIF" bash -c "
  source '$VENV/bin/activate'
  MODEL_PATH='$MODEL_PATH' GPU_MEM_UTIL=0.6 bash run_polymath_pipeline.sh
"
```

**Do not set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (or `PYTORCH_ALLOC_CONF`).**
It looks like a reasonable fragmentation fix, but on this cluster it's what was actually causing
every CUDA init failure — see the troubleshooting notes below.

Submit and monitor from the `evaluation/` directory:

```bash
cd evaluation
sbatch run_dias_polymath.sbatch
squeue -u $USER
tail -f polymath_<jobid>.out
```

If it OOMs for a genuine memory reason, `run_polymath_pipeline.sh` reads its memory/batch
settings from env vars, so you can override them without editing the pipeline itself — e.g. the
repo's README documents this conservative fallback:

```bash
apptainer exec --nv "$SIF" bash -c "
  source '$VENV/bin/activate'
  MODEL_PATH='$MODEL_PATH' GPU_MEM_UTIL=0.35 MAX_MODEL_LEN=2048 MAX_TOKENS=512 \
    MAX_NUM_SEQS=1 MAX_NUM_BATCHED_TOKENS=512 \
    EXTRA_GEN_ARGS='--enforce_eager --disable_prefix_caching' \
    bash run_polymath_pipeline.sh
"
```

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
is installed there), just re-run the setup commands from step 1 — `apptainer pull` + `venv` +
`pip install` (plus `git-lfs` if it's not already installed there either) — rather than copying
`$HOME/containers/python311.sif` and `$HOME/mule_venv` over. Reinstalling is quick (small base
image, pure pip installs) and more reliable than transferring a venv, whose paths are baked in
at creation and only work if the new machine's paths line up exactly.

## Troubleshooting notes (what actually mattered)

Getting this working took a long debugging session. In case this breaks again, or you're
setting this up somewhere else, here's what the real fixes were vs. what turned out to be red
herrings:

**Real fixes (keep these):**
- **`GPU` partition, not `LIGHTGPU`** — MIG slicing produces a `CUDA_VISIBLE_DEVICES` this
  vLLM version can't parse.
- **Non-slim `python:3.11` image** — vLLM/Triton need a C compiler (`gcc`) at runtime;
  `python:3.11-slim` doesn't have one.
- **`git-lfs pull`** — without it, the PolyMath dataset is just pointer-file stubs.
- **Do NOT set `PYTORCH_CUDA_ALLOC_CONF`/`PYTORCH_ALLOC_CONF=expandable_segments:True`** — this
  was the actual cause of every `CUDA driver error: out of memory` / `torch.OutOfMemoryError`
  crash, despite `nvidia-smi` always showing the GPU almost entirely free. It was added early on
  as an attempted fragmentation fix and turned out to be the opposite of helpful. Confirmed via
  isolated bisection: identical settings without this one env var work every time; with it, they
  fail unpredictably (sometimes instantly at CUDA device init, sometimes partway through model
  runner setup).

**Turned out not to matter (harmless to keep, but weren't the fix):**
- `vllm==0.16.0` pin — 0.17.1 failed with the exact same signature; 0.16.0 is just the version
  that was on hand when the real fix (removing `expandable_segments`) was found. It hasn't been
  re-tested whether 0.17.1 also works once `expandable_segments` is removed.
- `--mem=128G` (raised from the default 32G) — helped the failure surface slightly later in
  initialization, but didn't fix it outright. No reason to lower it back, though.
- `ulimit -l unlimited` — raised from a default of 64KB on a hunch (NCCL/pinned-memory
  registration), confirmed to apply, but made no difference to the crash.
- `flashinfer-python` uninstall — a plausible lead (the script's own comments warn about
  FlashInfer needing `nvcc`), but removing it didn't change the failure pattern either.
- `VLLM_USE_V1=0` — silently ignored; this vLLM version has removed the old V0 engine entirely.

**How it was actually diagnosed:** writing a minimal standalone script (`LLM(model=...)` +
one `generate()` call, no custom env vars or tuning) and confirming it worked, then
incrementally adding back the real pipeline's settings one group at a time (constructor kwargs,
env vars, import order) until something broke. Everything succeeded in isolation until
`expandable_segments` was added — that isolated test is what pinned it down, after many rounds
of changing one thing at a time in the full pipeline hadn't converged (in hindsight, because two
independent bugs — `expandable_segments` and the Git LFS stubs — were compounding, so fixing one
without the other looked like "no effect").