"""Fetch third-party robot assets at a pinned commit (never committed to git).

Franka Emika Panda from google-deepmind/mujoco_menagerie, Apache-2.0.
Usage:  python scripts/fetch_assets.py
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO = "https://github.com/google-deepmind/mujoco_menagerie.git"
COMMIT = "f3b6013629c7b94f8ee93fd977b161fa8350f9bc"   # fetched 2026-10-23
SUBDIR = "franka_emika_panda"
DEST = Path(__file__).resolve().parents[1] / "sim/assets/third_party" / SUBDIR
LOCK = Path(__file__).resolve().parents[1] / "sim/assets/assets.lock"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    expected = dict(line.split() for line in LOCK.read_text().splitlines() if line.strip() and not line.startswith("#"))
    if DEST.exists() and sha256(DEST / "panda.xml") == expected["panda.xml"]:
        print(f"{DEST} already present and verified")
        return
    with tempfile.TemporaryDirectory() as tmp:
        run = lambda *a: subprocess.run(a, cwd=tmp, check=True)
        run("git", "init", "-q")
        run("git", "remote", "add", "origin", REPO)
        run("git", "sparse-checkout", "set", SUBDIR)
        run("git", "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", COMMIT)
        run("git", "checkout", "-q", "FETCH_HEAD")
        src = Path(tmp) / SUBDIR
        got = sha256(src / "panda.xml")
        if got != expected["panda.xml"]:
            raise SystemExit(f"panda.xml sha256 mismatch: {got} != {expected['panda.xml']}")
        if DEST.exists():
            shutil.rmtree(DEST)
        DEST.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, DEST)
    print(f"fetched {SUBDIR}@{COMMIT[:8]} -> {DEST}")


if __name__ == "__main__":
    main()
