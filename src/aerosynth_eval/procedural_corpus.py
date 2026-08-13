"""Deterministic procedural baseline imagery for the frozen v0.1 scenarios.

The renderer creates deliberately simple, synthetic visual test assets. It is
not a photorealistic simulator and must not be used for operational inspection
or defect-diagnosis decisions.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from random import Random

from PIL import Image, ImageDraw, ImageFilter
from PIL import __version__ as PILLOW_VERSION
from pydantic import BaseModel, ConfigDict, Field

from aerosynth_eval.asset_registry import (
    AssetRegistryRecord,
    load_asset_registry,
    validate_asset_registry,
    write_asset_registry,
)
from aerosynth_eval.contracts import AssetLifecycleStatus, SurfaceCondition
from aerosynth_eval.scenario_matrix import ScenarioMatrixRecord, load_scenario_matrix

CANVAS_SIZE = (384, 256)
GENERATOR_NAME = "AeroSynth-Eval procedural renderer"
GENERATOR_MODEL = "surface-panel-renderer"
GENERATOR_VERSION = f"0.1.0+pillow-{PILLOW_VERSION}"
SOURCE_NOTE = (
    "Programmatically rendered procedural visual baseline; not an operational "
    "inspection image or defect-detection ground truth."
)


class ProceduralCorpusSummary(BaseModel):
    """Result of materializing the deterministic v0.1 image corpus."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_count: int = Field(ge=1)
    generated_now: int = Field(ge=0)
    reused_existing: int = Field(ge=0)
    asset_root: str = Field(min_length=1)


def _seed_for(scenario_id: str) -> int:
    """Derive a stable renderer seed from a frozen scenario identifier."""

    return int.from_bytes(hashlib.sha256(scenario_id.encode("utf-8")).digest()[:8], "big")


def _generation_prompt(scenario: ScenarioMatrixRecord) -> str:
    """Create a transparent, human-readable procedural rendering specification."""

    return (
        "Render a procedural synthetic aircraft exterior panel for a research "
        f"benchmark: component={scenario.component}, condition={scenario.condition}, "
        f"viewpoint={scenario.viewpoint}, lighting={scenario.lighting}, "
        f"quality_challenge={scenario.quality_challenge}. "
        "This is an illustrative visual baseline, not a photorealistic inspection image."
    )


def _surface_polygon(scenario: ScenarioMatrixRecord) -> list[tuple[int, int]]:
    """Return a component-specific panel silhouette."""

    if scenario.component.value == "fuselage_panel":
        return [(32, 52), (348, 43), (366, 198), (45, 211)]
    if scenario.component.value == "wing_panel":
        return [(22, 202), (79, 67), (355, 84), (325, 203)]
    return [(48, 203), (117, 54), (331, 84), (354, 198)]


def _draw_texture(
    image: Image.Image,
    polygon: list[tuple[int, int]],
    rng: Random,
) -> None:
    """Add lightweight panel texture, seams, and rivet-like details."""

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    min_x = min(x for x, _ in polygon)
    max_x = max(x for x, _ in polygon)
    min_y = min(y for _, y in polygon)
    max_y = max(y for _, y in polygon)

    for _ in range(520):
        x = rng.randint(min_x, max_x)
        y = rng.randint(min_y, max_y)
        alpha = rng.randint(8, 26)
        draw.point((x, y), fill=(18, 27, 35, alpha))

    draw.line([polygon[0], polygon[1]], fill=(238, 244, 248, 105), width=2)
    draw.line([polygon[-1], polygon[0]], fill=(18, 27, 35, 85), width=2)
    for fraction in range(12, 90, 8):
        x = min_x + ((max_x - min_x) * fraction // 100)
        y = min_y + ((max_y - min_y) * (100 - fraction) // 100)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(45, 55, 65, 125))
    image.alpha_composite(overlay)


def _draw_condition(
    image: Image.Image,
    scenario: ScenarioMatrixRecord,
    rng: Random,
) -> None:
    """Draw the requested surface condition without implying physical realism."""

    if scenario.condition is SurfaceCondition.NO_VISIBLE_DEFECT:
        return

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    center_x = rng.randint(145, 242)
    center_y = rng.randint(106, 165)

    if scenario.condition is SurfaceCondition.CORROSION:
        for _ in range(18):
            radius = rng.randint(4, 12)
            x = center_x + rng.randint(-42, 42)
            y = center_y + rng.randint(-28, 28)
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=(143, 79, 42, rng.randint(100, 175)),
                outline=(84, 48, 30, 190),
                width=1,
            )
    elif scenario.condition is SurfaceCondition.SURFACE_CRACK:
        points = [(center_x - 62, center_y + 17)]
        for step in range(1, 11):
            points.append(
                (
                    center_x - 62 + (step * 13),
                    center_y + 17 + rng.randint(-20, 20),
                )
            )
        draw.line(points, fill=(31, 23, 19, 245), width=3, joint="curve")
        draw.line(points, fill=(210, 200, 186, 130), width=1, joint="curve")
    elif scenario.condition is SurfaceCondition.COATING_DAMAGE:
        points = [
            (
                center_x + rng.randint(-46, 46),
                center_y + rng.randint(-32, 32),
            )
            for _ in range(11)
        ]
        draw.polygon(points, fill=(93, 102, 104, 220), outline=(44, 52, 57, 235), width=2)
        for _ in range(10):
            x = center_x + rng.randint(-33, 33)
            y = center_y + rng.randint(-24, 24)
            draw.line((x - 5, y - 3, x + 5, y + 3), fill=(190, 197, 195, 130), width=1)

    image.alpha_composite(overlay)


def _apply_capture_profile(
    image: Image.Image,
    scenario: ScenarioMatrixRecord,
) -> Image.Image:
    """Apply controlled lighting and quality challenges from the scenario matrix."""

    if scenario.capture_profile == "oblique_directional":
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.polygon(
            [(0, 0), (250, 0), (384, 256), (110, 256)],
            fill=(255, 225, 164, 42),
        )
        draw.polygon(
            [(260, 0), (384, 0), (384, 256), (150, 256)],
            fill=(13, 22, 33, 38),
        )
        image.alpha_composite(overlay)
    elif scenario.capture_profile == "close_glare":
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.ellipse((190, 44, 372, 176), fill=(255, 255, 242, 105))
        draw.ellipse((218, 66, 346, 154), fill=(255, 255, 255, 95))
        image.alpha_composite(overlay)
    elif scenario.capture_profile == "oblique_blur":
        return image.filter(ImageFilter.GaussianBlur(radius=1.1))
    return image


def render_procedural_asset(scenario: ScenarioMatrixRecord) -> tuple[bytes, int]:
    """Render one deterministic PNG asset and return its bytes and stable seed."""

    seed = _seed_for(scenario.scenario_id)
    rng = Random(seed)
    image = Image.new("RGBA", CANVAS_SIZE, (25, 34, 45, 255))
    background = ImageDraw.Draw(image)
    for y in range(CANVAS_SIZE[1]):
        shade = 34 + (y * 24 // CANVAS_SIZE[1])
        background.line((0, y, CANVAS_SIZE[0], y), fill=(shade, shade + 9, shade + 17, 255))

    polygon = _surface_polygon(scenario)
    surface_colours = {
        "fuselage_panel": (173, 185, 194, 255),
        "wing_panel": (160, 174, 184, 255),
        "empennage_panel": (181, 191, 197, 255),
    }
    ImageDraw.Draw(image).polygon(
        polygon,
        fill=surface_colours[scenario.component.value],
        outline=(82, 96, 107, 255),
        width=3,
    )
    _draw_texture(image, polygon, rng)
    _draw_condition(image, scenario, rng)
    rendered = _apply_capture_profile(image, scenario)

    buffer = BytesIO()
    rendered.convert("RGB").save(buffer, format="PNG", optimize=False)
    return buffer.getvalue(), seed


def _asset_path(asset_root: Path, image_reference: str) -> Path:
    """Resolve a validated registry reference below an explicit asset root."""

    reference = Path(image_reference)
    if reference.is_absolute() or ".." in reference.parts:
        raise ValueError(f"Unsafe image reference '{image_reference}'.")
    return asset_root / reference


def _generated_record(
    record: AssetRegistryRecord,
    scenario: ScenarioMatrixRecord,
    seed: int,
    image_sha256: str,
    generated_at: datetime,
) -> AssetRegistryRecord:
    """Attach complete, transparent provenance to a newly rendered asset."""

    payload = record.model_dump(mode="json")
    payload.update(
        {
            "lifecycle_status": AssetLifecycleStatus.GENERATED,
            "generator_name": GENERATOR_NAME,
            "generator_model": GENERATOR_MODEL,
            "generator_version": GENERATOR_VERSION,
            "generation_prompt": _generation_prompt(scenario),
            "seed": seed,
            "generated_at": generated_at.isoformat(),
            "image_sha256": image_sha256,
            "source_note": SOURCE_NOTE,
        }
    )
    return AssetRegistryRecord.model_validate(payload)


def materialize_procedural_corpus(
    registry_path: Path,
    scenario_matrix_path: Path,
    asset_root: Path,
) -> ProceduralCorpusSummary:
    """Render all planned v0.1 records and replace them with generated provenance."""

    records = load_asset_registry(registry_path)
    scenarios = load_scenario_matrix(scenario_matrix_path)
    validate_asset_registry(records, scenarios)
    scenarios_by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    generated_at = datetime.now(UTC).replace(microsecond=0)
    updated_records: list[AssetRegistryRecord] = []
    generated_now = 0
    reused_existing = 0

    for record in records:
        if record.lifecycle_status is AssetLifecycleStatus.REJECTED:
            raise ValueError("Procedural corpus materialization cannot include rejected assets.")

        scenario = scenarios_by_id[record.scenario_id]
        rendered, seed = render_procedural_asset(scenario)
        digest = hashlib.sha256(rendered).hexdigest()
        asset_path = _asset_path(asset_root, record.image_reference)

        if record.lifecycle_status is AssetLifecycleStatus.GENERATED:
            if record.seed != seed or record.image_sha256 != digest:
                raise ValueError(
                    f"Generated registry record '{record.asset_id}' does not match the "
                    "current deterministic renderer."
                )
            if asset_path.exists() and asset_path.read_bytes() != rendered:
                raise ValueError(
                    f"Existing asset '{asset_path}' does not match its generated provenance."
                )
            asset_path.parent.mkdir(parents=True, exist_ok=True)
            if not asset_path.exists():
                asset_path.write_bytes(rendered)
            updated_records.append(record)
            reused_existing += 1
            continue

        if asset_path.exists() and asset_path.read_bytes() != rendered:
            raise ValueError(f"Refusing to overwrite unexpected existing asset '{asset_path}'.")
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        asset_path.write_bytes(rendered)
        updated_records.append(_generated_record(record, scenario, seed, digest, generated_at))
        generated_now += 1

    write_asset_registry(registry_path, tuple(updated_records))
    return ProceduralCorpusSummary(
        record_count=len(updated_records),
        generated_now=generated_now,
        reused_existing=reused_existing,
        asset_root=str(asset_root),
    )
