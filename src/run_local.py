"""Find the organiser files already on this computer and write predictions.csv.

Looks in this repository's docs folder and in the user's Downloads folder.
blocks files are ignored. Train and test are separated by the label file.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_all import run  # noqa: E402


def search_dirs() -> list[Path]:
    home = Path.home()
    return [ROOT / "docs", home / "Downloads", home / "Desktop"]


def _skip(path: Path) -> bool:
    name = path.name.lower()
    if "block" in name or ".part" in name or "prediction" in name:
        return True
    if not any(token in name for token in ("transaction", "transfer", "label", "market_rate")):
        return True
    return any(part in {".git", "parts", "__pycache__", ".venv"} for part in path.parts)


def _useful_zip(path: Path) -> bool:
    name = path.name.lower()
    return any(token in name for token in ("decode", "detect", "hackathon", "training-data", "test-data"))


def collect_files() -> tuple[list[Path], list[Path]]:
    found = []
    archives = []
    for folder in search_dirs():
        if not folder.is_dir():
            continue
        for path in folder.rglob("*"):
            if not path.is_file() or _skip(path):
                continue
            suffix = path.suffix.lower()
            if suffix in {".parquet", ".csv"}:
                found.append(path)
            elif suffix == ".zip" and path.stat().st_size > 100_000 and _useful_zip(path):
                archives.append(path)
    # Prefer a larger file when two downloads share a stem, so the training
    # copy is not overwritten by the smaller test copy.
    found.sort(key=lambda path: path.stat().st_size, reverse=True)
    return found, archives


def _place(path: Path, docs: Path, used_names: set[str], seen_sizes: set[int]) -> None:
    size = path.stat().st_size
    if size in seen_sizes:
        return
    seen_sizes.add(size)
    target = docs / path.name
    if target.name in used_names:
        target = docs / f"{path.stem}_{size}{path.suffix}"
    used_names.add(target.name)
    shutil.copy(path, target)


def _extract_zip(archive: Path, docs: Path, used_names: set[str], seen_sizes: set[int]) -> None:
    try:
        opened = zipfile.ZipFile(archive)
    except zipfile.BadZipFile:
        return
    with opened:
        for info in opened.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            lower = name.lower()
            if _skip(Path(name)) or not lower.endswith((".parquet", ".csv")):
                continue
            if info.file_size in seen_sizes:
                continue
            seen_sizes.add(info.file_size)
            target = docs / f"{archive.stem}_{name}"
            if target.name in used_names:
                continue
            used_names.add(target.name)
            with opened.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            print(f"  unzipped {info.filename} from {archive.name}")


def main() -> int:
    files, archives = collect_files()
    print("files found:")
    for path in files:
        print(f"  {path}  ({path.stat().st_size:,} bytes)")
    if not files and not archives:
        raise SystemExit("No parquet or csv files were found in Downloads or docs.")

    with tempfile.TemporaryDirectory() as temporary:
        docs = Path(temporary) / "docs"
        docs.mkdir()
        used_names: set[str] = set()
        seen_sizes: set[int] = set()
        for path in files:
            _place(path, docs, used_names, seen_sizes)
        for archive in archives:
            _extract_zip(archive, docs, used_names, seen_sizes)
        print("training the models now. leave this window open.")
        destination = run(Path(temporary), fast=False, contest=True)
        # The temp folder is deleted when this block ends, so copy first.
        desktop_dir = Path.home() / "Desktop"
        desktop_dir.mkdir(parents=True, exist_ok=True)
        desktop = desktop_dir / "predictions.csv"
        local = ROOT / "submission" / "predictions.csv"
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(destination, desktop)
        shutil.copy(destination, local)
        print(destination.read_text(encoding="utf-8").splitlines()[0])
        print(f"predictions saved to {desktop}")
        print(f"predictions saved to {local}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
