#!/usr/bin/env python3
"""Fail CI if generated data/model artifacts are tracked."""

import fnmatch
import subprocess
import sys


BLOCKED_PATTERNS = (
    "*.sqlite",
    "*.sqlite3",
    "*.sqlite-journal",
    "*.sqlite-wal",
    "*.sqlite-shm",
    "*.parquet",
    "*.pt",
    "*.pkl",
    "src/data/features/cache/*",
    "src/data/models/saved_models/*",
    "src/data/models/saved_hyperparameters/*",
    "src/data/models/calibrators/*",
    "src/data/models/plots/*",
    "src/data/plots/*",
    "src/data/logs/*",
)


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def is_blocked(path: str) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in BLOCKED_PATTERNS)


def main() -> int:
    violations = [path for path in tracked_files() if is_blocked(path)]
    if not violations:
        print("No tracked generated artifacts found.")
        return 0

    print("Tracked generated artifacts are not allowed:")
    for path in violations:
        print(f"  - {path}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
