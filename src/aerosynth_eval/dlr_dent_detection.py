"""Safe indexing and capture-session splitting for the DLR dent release."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from zipfile import ZipFile


@dataclass(frozen=True)
class DlrDentRecord:
    """One image and its optional YOLO dent annotation inside the release ZIP."""

    image_id: str
    source_split: str
    image_member: str
    mask_member: str | None
    label_member: str | None
    boxes: tuple[tuple[int, float, float, float, float], ...]

    @property
    def has_dent(self) -> bool:
        return bool(self.boxes)

    @property
    def timestamp(self) -> int:
        return int(self.image_id)


def _image_id(filename: str, suffix: str) -> str:
    if not filename.endswith(suffix):
        raise ValueError(f"Unexpected DLR filename: {filename!r}")
    image_id = filename.removesuffix(suffix)
    if not image_id.isdigit():
        raise ValueError(f"DLR image ID is not a numeric capture timestamp: {filename!r}")
    return image_id


def _parse_yolo(payload: str, member: str) -> tuple[tuple[int, float, float, float, float], ...]:
    boxes: list[tuple[int, float, float, float, float]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{member}:{line_number} is not a five-field YOLO annotation")
        try:
            class_id = int(fields[0])
            coordinates = tuple(float(value) for value in fields[1:])
        except ValueError as error:
            raise ValueError(f"{member}:{line_number} has a non-numeric YOLO value") from error
        if class_id != 0 or any(value < 0.0 or value > 1.0 for value in coordinates):
            raise ValueError(f"{member}:{line_number} has an invalid normalized dent box")
        boxes.append((class_id, *coordinates))
    return tuple(boxes)


def scan_archive(path: str) -> list[DlrDentRecord]:
    """Map DLR image, mask, and YOLO labels without extracting source images.

    Images with no label file are retained as released negative examples.  An
    empty label file is also treated as negative, as in the YOLO convention.
    """

    images: dict[tuple[str, str], str] = {}
    masks: dict[tuple[str, str], str] = {}
    labels: dict[tuple[str, str], str] = {}
    label_payloads: dict[tuple[str, str], str] = {}
    with ZipFile(path) as archive:
        for member in archive.namelist():
            parts = PurePosixPath(member).parts
            if len(parts) != 4 or parts[1] not in {"train", "val", "test"}:
                continue
            split, directory, filename = parts[1:]
            key: tuple[str, str]
            if directory == "images" and filename.endswith("-image.png"):
                key = (split, _image_id(filename, "-image.png"))
                images[key] = member
            elif directory == "masks" and filename.endswith("-masks.png"):
                key = (split, _image_id(filename, "-masks.png"))
                masks[key] = member
            elif directory == "labels" and filename.endswith("-image.txt"):
                key = (split, _image_id(filename, "-image.txt"))
                labels[key] = member
                label_payloads[key] = archive.read(member).decode("utf-8")

    extra_labels = labels.keys() - images.keys()
    extra_masks = masks.keys() - images.keys()
    if extra_labels or extra_masks:
        raise ValueError("DLR archive contains labels or masks without their corresponding image")
    if set(images) != set(masks):
        raise ValueError("Every DLR image must have a matching released mask")

    records = [
        DlrDentRecord(
            image_id=image_id,
            source_split=split,
            image_member=member,
            mask_member=masks[(split, image_id)],
            label_member=labels.get((split, image_id)),
            boxes=_parse_yolo(label_payloads[(split, image_id)], labels[(split, image_id)])
            if (split, image_id) in labels
            else (),
        )
        for (split, image_id), member in images.items()
    ]
    if not records:
        raise ValueError("No DLR images were found in the expected release layout")
    return sorted(records, key=lambda record: (record.timestamp, record.source_split))


def session_groups(records: list[DlrDentRecord], gap_seconds: int) -> dict[str, str]:
    """Assign chronological capture sessions, separated by a recording gap."""

    if gap_seconds < 1:
        raise ValueError("gap_seconds must be positive")
    groups: dict[str, str] = {}
    previous: int | None = None
    group_number = 0
    for record in sorted(records, key=lambda item: (item.timestamp, item.source_split)):
        if previous is None or record.timestamp - previous > gap_seconds:
            group_number += 1
        groups[record.image_member] = f"session-{group_number:04d}"
        previous = record.timestamp
    return groups


def audit_capture_leakage(records: list[DlrDentRecord], gap_seconds: int) -> dict[str, int]:
    """Count provided-split session overlap; nonzero overlap means re-splitting is needed."""

    groups = session_groups(records, gap_seconds)
    split_by_group: dict[str, set[str]] = {}
    for record in records:
        split_by_group.setdefault(groups[record.image_member], set()).add(record.source_split)
    return {
        "gap_seconds": gap_seconds,
        "capture_sessions": len(split_by_group),
        "cross_source_split_sessions": sum(len(splits) > 1 for splits in split_by_group.values()),
    }
