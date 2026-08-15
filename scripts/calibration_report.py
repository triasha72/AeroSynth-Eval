from __future__ import annotations

import argparse
import json
from pathlib import Path

from aerosynth_eval.calibration import CalibrationPoint, fit_temperature, summarize_calibration

parser = argparse.ArgumentParser()
parser.add_argument("points", type=Path)
args = parser.parse_args()
points = [
    CalibrationPoint.model_validate(json.loads(line))
    for line in args.points.read_text().splitlines()
    if line.strip()
]
print(
    json.dumps(
        {
            "raw": summarize_calibration(points).model_dump(mode="json"),
            "fitted_temperature": fit_temperature(points),
        },
        indent=2,
    )
)
