#!/usr/bin/env python3
"""Create a text-free provenance receipt for the DLR aircraft-dent dataset."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from aerosynth_eval.dlr_dent import RECORD_ID, audit_record, source_digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.record_json is None:
        url = f"https://zenodo.org/api/records/{RECORD_ID}"
        with urllib.request.urlopen(url) as response:
            payload = response.read()
    else:
        payload = args.record_json.read_bytes()
    result = audit_record(json.loads(payload))
    result["zenodo_metadata_sha256"] = source_digest(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
