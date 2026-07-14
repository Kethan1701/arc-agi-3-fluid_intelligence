#!/usr/bin/env python
"""
bundle_for_kaggle.py
=====================
Builds `hybrid_lib` into a wheel for upload as a private Kaggle Dataset.

Why a wheel and not a code zip: verified against the official
`arcprize/ARC-AGI-3-Kaggle-Starter` repo's `scripts/build_notebook.py` --
the Kaggle submission notebook is built by splicing ONLY the raw text of
`agent/my_agent.py` into a notebook cell. Sibling files (this repo's
`hybrid_lib/` package) are never copied along. So `hybrid_lib` has to
reach the Kaggle notebook the same way the `arc-agi` package itself
does: as a wheel, installed offline in the notebook's install cell via
`pip install --no-index --find-links`.

This script does NOT touch `agent/my_agent.py` at all -- that file is
meant to be copied directly into your official ARC-AGI-3-Kaggle-Starter
clone (see README.md "Two-repo workflow"), where the starter's own
`scripts/build_notebook.py` handles splicing it into the notebook.

Usage:
    python scripts/bundle_for_kaggle.py [--out-dir dist]

Then, one-time per model-code change:
    1. Upload dist/*.whl as a new version of a private Kaggle Dataset
       (Kaggle web UI, or `kaggle datasets version -p dist/`).
    2. In your Kaggle-Starter clone's notebook install cell, add:
         !pip install --no-index --find-links \\
             /kaggle/input/<your-hybrid-lib-dataset>/ hybrid-lib
       alongside the existing `arc-agi` install line.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_wheel(out_dir: Path) -> list[Path]:
    if out_dir.exists():
        shutil.rmtree(out_dir)

    result = subprocess.run(
        ["uv", "build", "--out-dir", str(out_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout, file=sys.stdout)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(
            "uv build failed -- is `uv` installed and on PATH? "
            "(pip users: `pip install build && python -m build --wheel "
            f"--outdir {out_dir}` is an equivalent fallback.)"
        )

    wheels = sorted(out_dir.glob("*.whl"))
    if not wheels:
        raise SystemExit(f"uv build reported success but no .whl found in {out_dir}")
    return wheels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="dist", help="wheel output directory")
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.out_dir
    wheels = build_wheel(out_dir)

    print("Built:")
    for wheel in wheels:
        print(f"  {wheel}")
    print(
        "\nNext steps:\n"
        "  1. Upload the .whl above as a private Kaggle Dataset\n"
        "     (or a new version of an existing one).\n"
        "  2. In your ARC-AGI-3-Kaggle-Starter clone's notebook install cell,\n"
        "     add an offline install of it alongside the arc-agi wheel:\n"
        "       !pip install --no-index --find-links \\\n"
        "           /kaggle/input/<your-hybrid-lib-dataset>/ hybrid-lib\n"
        "  3. Copy agent/my_agent.py from THIS repo into that clone's\n"
        "     agent/my_agent.py, then `make submit` from there.\n"
        "\n"
        "Model weights are a SEPARATE Kaggle Dataset, not part of this\n"
        "wheel -- see README.md 'Uploading model weights'.\n"
    )


if __name__ == "__main__":
    main()
