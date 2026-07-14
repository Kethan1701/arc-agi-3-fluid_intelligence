"""
model_runtime.py
=================
Offline-only vLLM/PyTorch inference wrapper.

Kaggle's ARC-AGI-3 evaluation environment has *no internet access*, so
every path here must resolve to a local directory (typically
`/kaggle/input/<your-model-dataset>/...` at submission time, or a local
`models/<name>` directory during development). We set the standard
HF "do not phone home" environment variables defensively, on import,
so a stray `from_pretrained("org/model")`-style call anywhere in the
dependency tree fails fast locally instead of silently hanging/erroring
only once it's already burning the 9-hour Kaggle clock.

Design notes:
    * vLLM is imported lazily inside `VLLMPolicy.__init__`, not at module
      import time. vLLM does not install/run on native Windows -- if
      you're developing on Windows (as opposed to WSL2 or the eventual
      Linux Kaggle host), importing this module must not itself crash.
      Local development on Windows should use `StubPolicy` (bottom of
      this file) so you can exercise the full agent loop -- state
      tracking, fallback routing, resource management -- without a real
      GPU/vLLM install. Swap to `VLLMPolicy` under WSL2 or on the actual
      Kaggle instance (both Linux + NVIDIA GPU).
    * The policy's job is narrow: given a rendered prompt, return a
      chosen action id *and* a scalar confidence in [0, 1]. How confidence
      is derived (self-reported by the model in a structured JSON reply,
      vs. log-prob-based) is left explicit and swappable via
      `confidence_strategy`.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Protocol

# --- Force fully offline behavior BEFORE any HF/vLLM import happens ----
# These must be set prior to importing `transformers`/`vllm` anywhere in
# the process, since they read the env at import time.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
# Avoid vLLM's usage-stats ping (network call) and its own hub lookups.
os.environ.setdefault("VLLM_DO_NOT_TRACK", "1")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")


def detect_gpu_arch() -> tuple[str, tuple[int, int] | None]:
    """
    Identify the GPU compute capability so `VLLMPolicy` can apply
    architecture-specific workarounds. Returns (label, capability_tuple);
    capability is None if no CUDA device is visible (e.g. local CPU dev).

    Blackwell workstation/consumer parts (RTX 6000 Pro, RTX 50-series)
    report compute capability 12.0 ("sm_120"). Datacenter Blackwell
    (B100/B200/GB200) reports 10.x. Both are recent enough (as of vLLM
    0.13+) to need the flashinfer workaround below -- see
    `VLLMPolicy.__init__`.
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return "no-torch", None
    if not torch.cuda.is_available():
        return "no-cuda-device", None
    major, minor = torch.cuda.get_device_capability(0)
    if major == 12:
        return "blackwell-workstation", (major, minor)
    if major == 10:
        return "blackwell-datacenter", (major, minor)
    if major == 9:
        return "hopper", (major, minor)
    if major == 8:
        return "ampere", (major, minor)
    return f"sm_{major}{minor}", (major, minor)


@dataclass
class PolicyOutput:
    action: str
    action_args: dict | None
    confidence: float          # in [0, 1]; low values route to fallback
    raw_text: str               # raw model output, kept for debugging/logs


class Policy(Protocol):
    def act(self, prompt: str, action_space: list[str]) -> PolicyOutput: ...


class VLLMPolicy:
    """
    Real inference backend. Loads a local model directory (no hub calls)
    via vLLM and asks the model to emit a small JSON action decision.

    `model_dir` must be a local path, e.g.:
        Kaggle:  /kaggle/input/<dataset-slug>/<model-subdir>
        Local:   ./models/<model-name>
    containing whatever vLLM/HF need to load fully offline (config.json,
    tokenizer files, safetensors weights, etc.) -- i.e. exactly what you'd
    get from `snapshot_download(..., local_dir=...)` while you still have
    internet, then upload as a private Kaggle dataset.

    Blackwell (RTX 6000 Pro / "sm_120") notes:
        * Native SM120 CUTLASS/NVFP4 kernels landed in vLLM 0.13.0 --
          pin `vllm>=0.13.0` (and a matching CUDA-13 torch build) when
          bundling wheels for this GPU. Older vLLM builds either fail to
          import on sm_120 or silently fall back to slow paths.
        * As of the vLLM versions available at the time this was written,
          flashinfer's JIT compilation is broken on sm_120 when paired
          with the CUDA-13 PyPI wheels (triggered by
          `--attention-backend flashinfer` or `--kv-cache-dtype
          fp8_e5m2`). We defensively force `VLLM_ATTENTION_BACKEND=
          FLASH_ATTN` whenever `detect_gpu_arch()` reports a Blackwell
          part, unless the caller already set the env var explicitly.
          Re-check this against current vLLM release notes before
          submitting -- this is exactly the kind of fast-moving
          hardware-support detail that goes stale quickly.
        * The RTX 6000 Pro's 96GB of VRAM gives a lot of headroom versus
          typical Kaggle GPUs (T4=16GB, P100=16GB) -- `gpu_memory_utilization`
          can stay conservative (0.85 default here) since Kaggle notebooks
          also run non-GPU process overhead alongside vLLM, but you have
          plenty of room to raise `max_model_len` or batch size vs. what
          you'd use on a T4.
        * vLLM/torch are NOT preinstalled or required by the official
          ARC-AGI-3 Kaggle starter kit, and the accelerated Kaggle session
          has no internet access -- you must bundle matching vLLM/torch
          wheels (built for CUDA 13 / sm_120) as a private Kaggle Dataset
          and `pip install --no-index --find-links /kaggle/input/<wheels-
          dataset>/ vllm torch ...` in your submission notebook, the same
          offline-wheel pattern the starter kit already uses for the
          `arc-agi` package itself.
    """

    def __init__(
        self,
        model_dir: str,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.85,
        dtype: str = "bfloat16",
        max_new_tokens: int = 128,
    ) -> None:
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"model_dir={model_dir!r} does not exist locally. "
                "VLLMPolicy only loads local paths -- download the model "
                "while you have internet, then point this at the local "
                "copy (or the /kaggle/input/... dataset path at submit "
                "time)."
            )

        arch_label, capability = detect_gpu_arch()
        is_blackwell = arch_label in ("blackwell-workstation", "blackwell-datacenter")
        if is_blackwell and "VLLM_ATTENTION_BACKEND" not in os.environ:
            # See the Blackwell notes in this class's docstring: flashinfer
            # JIT is currently broken on sm_120 with CUDA-13 PyPI wheels.
            # Force the known-good backend rather than letting vLLM
            # auto-select and fail deep into a 9-hour run.
            os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"

        # Lazy import: keeps this module importable on machines without
        # vLLM installed (e.g. plain Windows dev boxes).
        from vllm import LLM, SamplingParams  # noqa: PLC0415

        import logging  # noqa: PLC0415

        logging.getLogger("arc_agent").info(
            "VLLMPolicy: detected GPU arch=%s capability=%s attention_backend=%s",
            arch_label, capability, os.environ.get("VLLM_ATTENTION_BACKEND", "<vllm default>"),
        )

        self.model_dir = model_dir
        self.max_new_tokens = max_new_tokens
        self.gpu_arch = arch_label

        self._llm = LLM(
            model=model_dir,
            tokenizer=model_dir,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            # Both flags below reinforce local-only loading at the vLLM
            # engine level, independent of the env vars set above.
            trust_remote_code=False,
            load_format="auto",
        )
        self._sampling_params = SamplingParams(
            temperature=0.2,
            max_tokens=max_new_tokens,
        )

    def act(self, prompt: str, action_space: list[str]) -> PolicyOutput:
        full_prompt = self._build_prompt(prompt, action_space)
        outputs = self._llm.generate([full_prompt], self._sampling_params)
        raw_text = outputs[0].outputs[0].text.strip()
        return self._parse(raw_text, action_space)

    def _build_prompt(self, situation_summary: str, action_space: list[str]) -> str:
        return (
            "You are playing an ARC-AGI-3 interactive grid puzzle.\n"
            f"Legal actions: {', '.join(action_space)}.\n"
            f"{situation_summary}\n"
            "Respond with ONLY a JSON object of the form "
            '{"action": "<one of the legal actions>", "confidence": <0..1 float>, '
            '"reason": "<short reason>"}.\n'
        )

    def _parse(self, raw_text: str, action_space: list[str]) -> PolicyOutput:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            # Model failed to follow the output format -- treat as
            # zero-confidence so the caller routes to the fallback solver
            # rather than crashing the run.
            return PolicyOutput(action=action_space[0], action_args=None, confidence=0.0, raw_text=raw_text)
        try:
            payload = json.loads(match.group(0))
            action = str(payload.get("action", ""))
            confidence = float(payload.get("confidence", 0.0))
        except (json.JSONDecodeError, TypeError, ValueError):
            return PolicyOutput(action=action_space[0], action_args=None, confidence=0.0, raw_text=raw_text)

        if action not in action_space:
            # Hallucinated an illegal action -- zero confidence, let the
            # fallback solver take over rather than sending a bad action.
            return PolicyOutput(action=action_space[0], action_args=None, confidence=0.0, raw_text=raw_text)

        confidence = max(0.0, min(1.0, confidence))
        return PolicyOutput(action=action, action_args=None, confidence=confidence, raw_text=raw_text)

    def shutdown(self) -> None:
        """Release the vLLM engine's GPU memory. See resource_manager.py."""
        import gc  # noqa: PLC0415

        del self._llm
        gc.collect()
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass


class StubPolicy:
    """
    No-GPU-required stand-in for `VLLMPolicy`, used for local development
    (especially on Windows, where vLLM isn't installable) and unit tests.
    Always reports zero confidence so the agent's routing logic always
    falls through to `FallbackSolver` -- this lets you fully exercise the
    agent loop, state tracking, and resource manager without a real model.
    """

    def act(self, prompt: str, action_space: list[str]) -> PolicyOutput:
        return PolicyOutput(
            action=action_space[0] if action_space else "NOOP",
            action_args=None,
            confidence=0.0,
            raw_text="[StubPolicy] no model loaded",
        )

    def shutdown(self) -> None:
        pass
