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

        # Lazy import: keeps this module importable on machines without
        # vLLM installed (e.g. plain Windows dev boxes).
        from vllm import LLM, SamplingParams  # noqa: PLC0415

        self.model_dir = model_dir
        self.max_new_tokens = max_new_tokens

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
