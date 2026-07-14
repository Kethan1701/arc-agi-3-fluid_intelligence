#!/usr/bin/env python
"""
bundle_for_kaggle.py
=====================
Packages this repo's code (NOT model weights) into a submission-ready
zip, mirroring the layout the Kaggle notebook/harness expects when it
imports `agent.my_agent`.

What gets bundled:
    agent/, env/, configs/  -- pure code + small config files.

What does NOT get bundled, on purpose:
    models/  -- model weights belong in a separate Kaggle Dataset
    attached to your submission notebook, not inside a code zip. Kaggle
    Datasets handle multi-GB files properly (versioning, mounting at
    /kaggle/input/<slug>/ read-only); a code-submission zip does not.
    This script will refuse to bundle anything under models/ and will
    print a reminder of the two-part submission shape:
        1. Code (this zip, or the notebook that imports it)
        2. Model weights (a Kaggle Dataset you create separately, see
           README.md "Uploading model weights" section)

Usage:
    python scripts/bundle_for_kaggle.py [--out-dir kaggle_submission] [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_DIRS = ["agent", "env", "configs"]


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def build_manifest(config_path: str) -> dict:
    return {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "config_used": config_path,
        "bundled_dirs": CODE_DIRS,
        "note": (
            "Model weights are NOT included in this zip. Upload them as a "
            "separate private Kaggle Dataset and set agent.model_dir to "
            "the mounted /kaggle/input/... path. See README.md."
        ),
    }


def stage_and_zip(out_dir: Path, config_path: str) -> Path:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    staging = out_dir / "staging"
    staging.mkdir(parents=True)

    for rel_dir in CODE_DIRS:
        src = REPO_ROOT / rel_dir
        if not src.exists():
            print(f"WARNING: expected directory {src} not found, skipping.", file=sys.stderr)
            continue
        dst = staging / rel_dir
        shutil.copytree(
            src,
            dst,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
        )

    manifest = build_manifest(config_path)
    (staging / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    zip_path = out_dir / "arc_agi3_submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in staging.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(staging))

    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="kaggle_submission", help="staging/output directory")
    parser.add_argument("--config", default="configs/default.yaml", help="config file to record in the manifest")
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.out_dir
    zip_path = stage_and_zip(out_dir, args.config)

    print(f"Bundled: {', '.join(CODE_DIRS)}")
    print(f"Wrote:   {zip_path}")
    print(
        "\nReminder: model weights are NOT in this zip.\n"
        "  1. Pre-download your model locally while you have internet.\n"
        "  2. Upload it as a private Kaggle Dataset.\n"
        "  3. Attach that dataset to your submission notebook and point\n"
        "     AgentConfig.model_dir at its /kaggle/input/... path.\n"
    )


if __name__ == "__main__":
    main()
