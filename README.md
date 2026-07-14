# arc-agi-3-fluid_intelligence

Local development pipeline and baseline agent for the Kaggle **ARC Prize
2026 — ARC-AGI-3** competition (interactive reasoning benchmark: agents
explore dynamic game environments, build a world model, and act
step-by-step, rather than transforming a static input grid).

## Directory structure

```
arc-agi-3-fluid_intelligence/
├── agent/
│   ├── my_agent.py         # ArcAgent: main decision loop + hybrid routing
│   ├── model_runtime.py    # VLLMPolicy (offline vLLM) + StubPolicy (no-GPU dev)
│   ├── state.py             # GameMemory (per-game) / SessionMemory (per-run)
│   ├── fallback_solver.py  # deterministic heuristic/exploration solver
│   └── resource_manager.py # 9h wall-clock budget + VRAM cleanup
├── env/
│   ├── env_adapter.py      # ArcEnv protocol + Kaggle harness shim (EDIT before submitting)
│   └── local_mock_env.py   # offline toy env for local dev/testing
├── configs/
│   └── default.yaml        # thresholds, model settings, resource budget
├── scripts/
│   ├── run_local.py        # play N episodes locally, print summary
│   └── bundle_for_kaggle.py# package agent/env/configs into a submission zip
├── tests/
│   └── test_agent_smoke.py # offline end-to-end smoke tests (pytest)
├── models/                 # local model weight dirs go here (gitignored)
├── requirements.txt
└── .gitignore
```

## Setup (local, Windows-friendly)

This machine already has `uv` on PATH, so:

```powershell
cd path\to\arc-agi-3-fluid_intelligence
uv venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```

(Plain `python -m venv .venv && pip install -r requirements.txt` also
works.) The default `requirements.txt` only installs CPU-only, GPU-free
packages — `torch`/`vllm` are commented out because **vLLM does not
install natively on Windows**. The full agent loop (state tracking,
fallback routing, resource management) is fully testable without them via
`agent.model_runtime.StubPolicy`, which is the default in
`configs/default.yaml` (`use_vllm: false`).

## Running the local smoke loop

```powershell
python scripts\run_local.py --episodes 10
```

This plays the agent (StubPolicy, i.e. always routes to the deterministic
fallback solver) against `LocalMockEnv` and prints a solve-rate + resource
summary. Use `--verbose` for per-step logs.

## Running tests

```powershell
pytest tests/
```

## Testing real inference (VLLMPolicy)

vLLM requires Linux + an NVIDIA GPU. Two options:

1. **WSL2**: install a Linux distro, install `torch`/`vllm` from the
   commented block in `requirements.txt` there, download a small local
   model, then run:
   ```bash
   python scripts/run_local.py --episodes 3 --use-vllm --model-dir ./models/your-model
   ```
2. **Kaggle notebook**: iterate directly in a GPU-enabled Kaggle notebook,
   which matches the actual submission environment (RTX 6000, offline).

`model_dir` must be a **local directory** (already downloaded — see
below). `VLLMPolicy` refuses to fall back to a hub download; internet is
disabled at Kaggle inference time, so any hub call would just hang/fail
there, and this scaffold is designed to fail loudly and immediately
instead, in local dev, if the path doesn't exist.

## Uploading model weights for Kaggle

Kaggle Datasets, not the code submission zip, are the right place for
model weights:

1. While you still have internet, download your model to a local folder:
   ```python
   from huggingface_hub import snapshot_download
   snapshot_download("org/model-name", local_dir="models/your-model")
   ```
2. Create a private Kaggle Dataset from that local folder (Kaggle web UI
   or `kaggle datasets create`).
3. Attach the dataset to your submission notebook — it mounts read-only
   at `/kaggle/input/<dataset-slug>/...`.
4. Point `AgentConfig.model_dir` (in `agent/my_agent.py::get_agent()`, or
   `configs/default.yaml`) at that mounted path.

## Packaging for submission

```powershell
python scripts\bundle_for_kaggle.py
```

Produces `kaggle_submission/arc_agi3_submission.zip` containing
`agent/`, `env/`, `configs/`, and a `MANIFEST.json` (git commit, config
used, timestamp). Model weights are **not** included — see above.

## Before your first real submission — two manual steps

This scaffold was built without internet access to the competition's
private starter kit, so two things are deliberately left as
placeholders rather than guessed at:

1. **Confirm the harness calling convention.** Once you've accepted the
   competition rules, download the official ARC-AGI-3 starter kit from
   the Kaggle competition page. Check how it actually invokes your
   submitted agent (module-level function? class with a specific
   method?) and reconcile that against `env/env_adapter.py`'s
   `KaggleHarnessEnv` shim and `agent/my_agent.py`'s `get_agent()`
   entrypoint — both are clearly marked with `EDIT ME` / docstring notes.
2. **Stage your model weights** as described above and update
   `model_dir` accordingly.

## Design notes

- **Hybrid routing** (`agent/my_agent.py::ArcAgent._decide`): resource
  budget exhausted → fallback; detected action loop → fallback; primary
  policy confidence below `confidence_threshold` → fallback; otherwise
  use the primary policy. This order means a healthy, confident model is
  always preferred, but the pipeline never blocks the full 9-hour budget
  on a model that's stuck, slow, or unavailable.
- **Confidence** is self-reported by the model in a structured JSON
  response (`{"action": ..., "confidence": ..., "reason": ...}`) rather
  than derived from token log-probs — simpler to reason about and
  model-agnostic, at the cost of trusting the model's self-assessment.
  If you find the model is overconfident/underconfident in practice,
  swap in a log-prob-based confidence estimator in `VLLMPolicy._parse`.
- **`FallbackSolver`** is deliberately simple (optimistic
  untried-action exploration, loop-breaking, effectiveness-weighted
  exploration) rather than a hand-built ARC-DSL search, since game rules
  are unknown a priori and vary per game. `_search_action_sequence` is
  a documented, unimplemented extension point for deeper search once
  you've characterized specific game families worth targeting.
