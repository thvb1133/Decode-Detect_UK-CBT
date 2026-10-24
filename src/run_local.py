"""Find the organiser files already on this computer and write predictions.csv.

Looks in this repository's docs folder and in the user's Downloads folder.
blocks files are ignored. Train and test are separated by the label file.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_all import run  # noqa: E402


def search_dirs() -> list[Path]:
    home = Path.home()
    return [ROOT / "docs", home / "Downloads", home / "Desktop"]


def collect_files() -> list[Path]:
    found = []
    for folder in search_dirs():
        if not folder.is_dir():
            continue
        for path in folder.glob("*"):
            if not path.is_file():
                continue
            name = path.name.lower()
            if path.suffix.lower() not in {".parquet", ".csv"}:
                continue
            if "block" in name or ".part" in name:
                continue
            found.append(path)
    # Prefer a larger file when two downloads share a stem, so the training
    # copy is not overwritten by the smaller test copy.
    found.sort(key=lambda path: path.stat().st_size, reverse=True)
    return found


def main() -> int:
    files = collect_files()
    print("files found:")
    for path in files:
        print(f"  {path}  ({path.stat().st_size:,} bytes)")
    if not files:
        raise SystemExit("No parquet or csv files were found in Downloads or docs.")

    with tempfile.TemporaryDirectory() as temporary:
        docs = Path(temporary) / "docs"
        docs.mkdir()
        used_names = set()
        for path in files:
            target = docs / path.name
            if target.name in used_names:
                target = docs / f"{path.stem}_{path.stat().st_size}{path.suffix}"
            used_names.add(target.name)
            shutil.copy(path, target)
        destination = run(Path(temporary), fast=False)

    desktop = Path.home() / "Desktop" / "predictions.csv"
    shutil.copy(destination, desktop)
    local = ROOT / "submission" / "predictions.csv"
    local.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(destination, local)
    print(destination.read_text(encoding="utf-8").splitlines()[0])
    print(f"predictions saved to {desktop}")
    print(f"predictions saved to {local}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
