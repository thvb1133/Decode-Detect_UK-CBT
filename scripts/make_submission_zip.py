"""Package the source tree into the code ZIP the organisers accept.

Only source, documentation, dependencies and the licence go in. Datasets,
caches, virtual environments and the predictions CSV are excluded: the CSV is
uploaded separately, and the archive limits are 8 MiB compressed, 64 MiB
uncompressed and 2,000 files.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

INCLUDE_DIRS = ("src", "tests", "scripts", "docs")
INCLUDE_FILES = ("README.md", "LICENSE", "requirements.txt")
EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", ".git", "cache", "data_raw", ".ipynb_checkpoints"}
EXCLUDE_SUFFIXES = {".parquet", ".pyc", ".zip", ".joblib"}

MAX_COMPRESSED = 8 * 1024 * 1024
MAX_UNCOMPRESSED = 64 * 1024 * 1024
MAX_FILES = 2000


def _wanted(path: Path) -> bool:
    if any(part in EXCLUDE_PARTS for part in path.parts):
        return False
    return path.suffix not in EXCLUDE_SUFFIXES


def build(root: Path, out: Path, metrics: Path | None = None):
    root = Path(root)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    members = []
    for name in INCLUDE_FILES:
        if (root / name).exists():
            members.append(root / name)
    for directory in INCLUDE_DIRS:
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and _wanted(path.relative_to(root)):
                members.append(path)
    if metrics and Path(metrics).exists():
        members.append(Path(metrics))

    # ZIP_DEFLATED only: encrypted, multi-part and ZIP64 archives are rejected
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=False) as archive:
        for path in members:
            archive.write(path, path.relative_to(root))

    uncompressed = sum(p.stat().st_size for p in members)
    compressed = out.stat().st_size
    print(f"files         {len(members)} (limit {MAX_FILES})")
    print(f"uncompressed  {uncompressed / 1024 / 1024:.2f} MiB (limit 64 MiB)")
    print(f"compressed    {compressed / 1024 / 1024:.2f} MiB (limit 8 MiB)")

    problems = []
    if len(members) > MAX_FILES:
        problems.append("too many files")
    if uncompressed > MAX_UNCOMPRESSED:
        problems.append("uncompressed size over limit")
    if compressed > MAX_COMPRESSED:
        problems.append("compressed size over limit")
    for path in members:
        if path.suffix == ".parquet" or "data_raw" in path.parts:
            problems.append(f"dataset leaked into the archive: {path}")
    if problems:
        print("\nFAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"\nPASSED: wrote {out}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="submission/code.zip")
    parser.add_argument("--metrics", default="artifacts/metrics.json")
    args = parser.parse_args(argv)
    raise SystemExit(build(Path(args.root), Path(args.out), Path(args.metrics)))


if __name__ == "__main__":
    main()
