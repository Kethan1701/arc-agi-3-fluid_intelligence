# arc-agi-3-fluid_intelligence

> ## ⚠️ The wheel/dataset workflow below is retired
>
> The shipped agent now lives **only** in the starter-kit clone, at
> `ARC-AGI-3-Kaggle-Starter/agent/my_agent.py`, and is entirely
> self-contained: standard library + `arcengine` + the framework's
> `Agent` base class. It does **not** import `hybrid_lib`.
>
> **Why the change.** `hybrid_lib` could only reach Kaggle as a wheel in
> a private Kaggle Dataset, and that plumbing caused both of the
> submission failures on this project: first a `--find-links` pointing at
> a `kaggle.com` URL, which cannot resolve because runtime internet is
> disabled (Kaggle reported it only as a bare "system error"), and then a
> follow-up fix that replaced the install cell wholesale and silently
> dropped the `arc-agi` install. Neither failure was about the agent's
> logic. Making the one shipped file self-contained deletes the private
> dataset, the `dataset_sources` attach, the second `pip install`, and
> the mount-path check — the entire failure mode — at the cost of nothing
> the submission was actually using.
>
> **Why the logic changed too.** `hybrid_lib`'s solver scored 0 on every
> game: it was capped at 80 actions per game, aimed `ACTION6` at
> uniformly random coordinates (1 of 4096, almost always background), and
> had no notion of level progress. The replacement maps the reachable
> state space as a graph and routes to unexplored frontiers, and aims
> clicks at detected objects. On the local dev games that took ls20 from
> 0 to level 1 and vc33 from 0 to level 2.
>
> `hybrid_lib/` and `scripts/bundle_for_kaggle.py` are kept here as
> reference, but nothing in the submission path reads them. Do not
> re-introduce the private-dataset install step.

`hybrid_lib`: a reusable, testable decision core (episodic memory +
deterministic fallback solver + confidence-gated LLM routing + resource
budget management) for a Kaggle **ARC Prize 2026 — ARC-AGI-3** agent,
plus a reference `agent/my_agent.py` that wires it into the real
competition harness contract.

ARC-AGI-3 is an interactive reasoning benchmark: agents explore dynamic
game environments, build a world model, and act step-by-step, rather
than transforming a static input grid like prior ARC competitions.

Everything here — the harness interface, `arcengine` types, the
Blackwell/RTX 6000 Pro GPU notes, the concurrency model — was verified
by cloning and reading the official `arcprize/ARC-AGI-3-Kaggle-Starter`
and `arcprize/ARC-AGI-3-Agents` repos directly, not guessed at.

## Two-repo workflow

This is important and not how the first draft of this project was
structured, so it's worth being explicit about: **your actual Kaggle
submission is built from a separate clone of the official starter kit,
not from this repo directly.**

Why: the official `scripts/build_notebook.py` (in
`arcprize/ARC-AGI-3-Kaggle-Starter`) builds your submission notebook by
splicing the **raw text of one file**, `agent/my_agent.py`, into a
notebook cell. It does not copy any sibling files. So this repo's
`hybrid_lib/` package — the actual engineering (memory, fallback solver,
resource management, offline vLLM wrapper) — has to reach Kaggle a
different way: as a wheel, installed offline in the notebook, exactly
the way the starter kit already installs the `arc-agi` package itself.

The split (the wheel row below is **superseded — see callout at top**;
`agent/my_agent.py` no longer imports `hybrid_lib`, so nothing installs it
on the Kaggle side anymore):

| This repo (`arc-agi-3-fluid_intelligence`)          | Your official starter clone                          |
|------------------------------------------------------|--------------------------------------------------------|
| `hybrid_lib/` — the reusable decision core, unit tested | `agent/my_agent.py` — the one file Kaggle reads, copied in from here |
| `agent/my_agent.py` — reference implementation to copy over | `make setup / play-local / submit / status` — the real local-dev and submission workflow |
| `scripts/bundle_for_kaggle.py` — builds the `hybrid_lib` wheel *(superseded)* | vendors `ARC-AGI-3-Agents` (MIT) and the live gateway connection for real local play |

We deliberately do **not** vendor the starter kit's Makefile/scripts
into this repo: as of this writing it has no LICENSE file (unlike
`ARC-AGI-3-Agents`, which is MIT), so we point you at cloning it
yourself rather than redistributing a copy here.

### Setting it up

```powershell
# 1. This repo: the hybrid_lib library + its tests.
cd path\to\arc-agi-3-fluid_intelligence
uv venv
uv pip install -e . -r requirements.txt --python .venv\Scripts\python.exe

# 2. A sibling clone of the official starter kit (needs its own
#    Python 3.12 + make; see its README for full setup, incl. an
#    ARC Prize API key for local play and a Kaggle API token for submit).
cd ..
git clone https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter.git
cd ARC-AGI-3-Kaggle-Starter
make setup

# 3. Copy the reference agent over, then point it at hybrid_lib:
cp ../arc-agi-3-fluid_intelligence/agent/my_agent.py agent/my_agent.py
.venv/bin/pip install -e ../arc-agi-3-fluid_intelligence   # editable, for local play-local

make play-local          # drives MyAgent against real games over the network
```

`make play-local` in the starter kit connects over HTTP to the live ARC
Prize game server (needs an API key and internet **on your dev
machine** — this is separate from, and does not violate, the
no-internet constraint that only applies to the actual Kaggle scoring
environment).

## Directory structure (this repo)

```
arc-agi-3-fluid_intelligence/
├── hybrid_lib/                  # installable package -- kept as reference only (superseded, see callout at top)
│   ├── decision_core.py         # HybridDecisionCore: shared, thread-safe routing brain
│   ├── model_runtime.py         # VLLMPolicy (offline vLLM, Blackwell-aware) + StubPolicy
│   ├── state.py                 # GameMemory (per-game) / SessionMemory (per-run)
│   ├── fallback_solver.py       # deterministic heuristic/exploration solver
│   └── resource_manager.py      # 9h wall-clock budget, thread-safe
├── agent/
│   └── my_agent.py              # MyAgent(Agent) reference impl -- copy into your starter clone
├── tests/
│   └── test_hybrid_lib.py       # offline tests against REAL arcengine types
├── models/                      # local model weight dirs go here (gitignored)
├── pyproject.toml               # hybrid_lib packaging (hatchling)
├── requirements.txt
└── .gitignore
```

## Running tests

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

Tests exercise `HybridDecisionCore` directly against real
`arcengine.FrameData`/`GameAction`/`GameState` objects (the `arcengine`
package is pip-installable standalone — pure Python + numpy + pydantic,
no GPU needed, works on Windows). They deliberately do **not**
instantiate `MyAgent` itself, since it subclasses `agents.agent.Agent`
from the separate `ARC-AGI-3-Agents` framework, which only becomes
importable once you've vendored that repo (step 2 above). One test
(`test_concurrent_games_do_not_corrupt_shared_state`) plays several toy
games on separate threads against one shared core, mirroring the real
harness's concurrency model (see below).

## Concurrency model — read this before writing GPU code

Verified in `agents/swarm.py` (`ARC-AGI-3-Agents`): the real harness
creates **one `MyAgent` instance per game** and runs all of them
**concurrently, each in its own thread, in one process** — with up to
~110 games in the full competition rerun, that's potentially ~110
threads calling `choose_action` around the same time, all sharing
whatever model you load.

Consequences already handled in `hybrid_lib`:
- `HybridDecisionCore` is a process-wide singleton
  (`hybrid_lib.decision_core.get_shared_core`), so the model loads once,
  not once per game.
- `ResourceManager` tracks per-game timing in a dict keyed by `game_id`
  (not a single shared attribute) and guards all mutation with a lock.
- `VLLMPolicy.act()` serializes concurrent calls with a lock, since
  vLLM's synchronous `LLM.generate()` isn't safe for unsynchronized
  concurrent calls from multiple threads. This is a **correctness**
  choice, not a throughput-optimal one — calls queue rather than being
  batched. If this is a bottleneck in practice, the real fix is vLLM's
  `AsyncLLMEngine` (or an OpenAI-compatible server alongside the
  notebook) so vLLM does the batching instead of this lock serializing
  everything.
- There is deliberately **no per-game VRAM cleanup** —
  `torch.cuda.empty_cache()` between games isn't safe when other games'
  threads may have in-flight requests against the same model. VRAM
  cleanup is a single `HybridDecisionCore.shutdown()` call, meant for
  process exit only, after every game thread has finished.

## Running real inference (VLLMPolicy) on RTX 6000 Pro (Blackwell)

Yes, this works — `rtx6000` (Kaggle accelerator `nvidiaRtx6000`, a GCP
`g4-standard-48` instance) is a real, first-class, ARC-AGI-3-exclusive
accelerator option, set via `ACCELERATOR` in your starter clone's
`scripts/build_notebook.py`. It burns GPU quota faster than T4/P100, so
the starter kit's own comment says "use only when you're confident."

Verified compatibility notes (re-check against current vLLM release
notes before submitting — this is a fast-moving surface):
- `vllm>=0.13.0` is the first release with native SM120 (Blackwell)
  CUTLASS/NVFP4 kernels.
- flashinfer's JIT compilation is currently broken on sm_120 with
  CUDA-13 PyPI wheels. `hybrid_lib/model_runtime.py`'s
  `detect_gpu_arch()` detects Blackwell and `VLLMPolicy` forces
  `VLLM_ATTENTION_BACKEND=FLASH_ATTN` automatically as a defensive
  workaround.
- **Neither torch nor vLLM is preinstalled or required by the official
  starter kit**, and Kaggle's accelerated sessions have zero internet
  access. You must bundle matching CUDA-13 torch/vllm wheels as a
  private Kaggle Dataset and install them offline in your submission
  notebook, the same `pip install --no-index --find-links` pattern the
  starter kit already uses for `arc-agi` itself. See the commented block
  in `requirements.txt`.
- The 96GB VRAM gives a lot of headroom vs. T4/P100 (16GB each) — plenty
  of room to raise `max_model_len` or batch size once you've profiled a
  baseline.

Since vLLM doesn't install on native Windows, test the real
`VLLMPolicy` path either in WSL2 or directly in a Kaggle notebook (both
Linux + NVIDIA GPU). `StubPolicy` (the local default) exercises every
other code path — memory, fallback routing, resource management,
concurrency — without needing a GPU at all.

## Uploading model weights for Kaggle

Kaggle Datasets, not any code bundle, are the right place for model
weights:

1. While you still have internet, download your model to a local folder:
   ```python
   from huggingface_hub import snapshot_download
   snapshot_download("org/model-name", local_dir="models/your-model")
   ```
2. Create a private Kaggle Dataset from that local folder.
3. Attach it to your submission notebook — mounts read-only at
   `/kaggle/input/<dataset-slug>/...`.
4. Point `CONFIG.model_dir` in `agent/my_agent.py` at that mounted path,
   and set `CONFIG.use_vllm = True`.

## Packaging hybrid_lib for Kaggle (superseded — see callout at top)

Kept for reference only; nothing in the submission path installs this wheel
anymore.

```powershell
python scripts\bundle_for_kaggle.py
```

Runs `uv build`, producing `dist/hybrid_lib-*.whl`. Upload that as a
private Kaggle Dataset (or a new version of an existing one), then add
an offline install line for it in your starter clone's notebook install
cell, alongside the `arc-agi` wheel install. Full steps are printed by
the script.

## Design notes

- **Hybrid routing** (`hybrid_lib/decision_core.py::HybridDecisionCore._decide`):
  resource budget exhausted → fallback; detected action loop → fallback;
  primary policy confidence below `confidence_threshold` → fallback;
  otherwise use the primary policy. A healthy, confident model is always
  preferred, but the pipeline never blocks the 9-hour budget on a model
  that's stuck, slow, or unavailable.
- **Confidence** is self-reported by the model in a structured JSON
  response (`{"action": ..., "confidence": ..., "reason": ...}`) rather
  than derived from token log-probs — simpler and model-agnostic, at the
  cost of trusting the model's self-assessment. Swap in a log-prob-based
  estimator in `VLLMPolicy._parse` if you find it under/overconfident.
- **`FallbackSolver`** is deliberately simple (optimistic untried-action
  exploration, loop-breaking, effectiveness-weighted exploration) rather
  than a hand-built ARC-DSL search, since game rules are unknown a
  priori and vary per game. `_search_action_sequence` is a documented,
  unimplemented extension point for deeper search once you've
  characterized specific game families (via real `make play-local` runs)
  worth targeting.
- **Frame shape**: `arcengine.FrameData.frame` is `list[list[list[int]]]`
  — a frame is a *list of 2D grids*, not one grid. `hybrid_lib.state`
  types this as `MultiGrid` and hashes/compares the whole thing.

## Before your first real submission

1. Clone the official starter kit (see "Two-repo workflow" above) and
   run through its own README — it needs an ARC Prize API key (for
   `make play-local`) and a Kaggle API token (for `make submit`).
2. ~~Build and upload the `hybrid_lib` wheel (above), and edit the install
   cell in the starter's notebook build script to install it.~~
   **Superseded — see callout at top.** `agent/my_agent.py` is
   self-contained now; skip this step.
3. Stage and upload your model weights as their own Kaggle Dataset, and
   set `CONFIG.model_dir`/`CONFIG.use_vllm` in `agent/my_agent.py`.
4. If you're targeting the RTX 6000 Pro accelerator, also bundle
   matching CUDA-13 torch/vllm wheels (see above) — nothing GPU-related
   is preinstalled.
5. `make play-local` in your starter clone to sanity check against real
   games before `make submit`.
