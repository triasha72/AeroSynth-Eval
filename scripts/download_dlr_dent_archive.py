#!/usr/bin/env python3
"""Download the DLR dent archive resumably after explicit confirmation."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from aerosynth_eval.dlr_dent import EXPECTED_FILE


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external/dlr-aircraft-dent") / EXPECTED_FILE,
    )
    parser.add_argument("--confirm-large-download", action="store_true")
    parser.add_argument("--minimum-free-gb", type=int, default=12)
    args = parser.parse_args()

    if not args.confirm_large_download:
        raise SystemExit("Refusing the 5.7 GB download; pass --confirm-large-download to continue.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(args.output.parent).free
    required_bytes = args.minimum_free_gb * 1024**3
    if free_bytes < required_bytes:
        raise SystemExit(
            f"Only {free_bytes / 1024**3:.1f} GiB free; need at least {args.minimum_free_gb} GiB."
        )
    subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--continue-at",
            "-",
            "--output",
            str(args.output),
            args.url,
        ],
        check=True,
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
