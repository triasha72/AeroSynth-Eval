"""Inspect the DLR dent archive without extracting its source images."""

from __future__ import annotations

from collections import Counter
from pathlib import PurePosixPath
from zipfile import ZipFile

IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
ANNOTATION_SUFFIXES = frozenset({".csv", ".json", ".txt", ".xml", ".yaml", ".yml"})


def inspect_archive(path: str) -> dict[str, object]:
    """Return a content-free inventory for a downloaded DLR release archive.

    This intentionally reads archive metadata only.  It does not extract images,
    annotations, or source text, so it is safe to commit the resulting receipt.
    """

    suffixes: Counter[str] = Counter()
    top_level: Counter[str] = Counter()
    image_paths: list[str] = []
    annotation_paths: list[str] = []

    with ZipFile(path) as archive:
        entries = [entry for entry in archive.infolist() if not entry.is_dir()]
        for entry in entries:
            member = PurePosixPath(entry.filename)
            suffix = member.suffix.casefold()
            suffixes[suffix or "[none]"] += 1
            top_level[member.parts[0] if member.parts else "[root]"] += 1
            if suffix in IMAGE_SUFFIXES:
                image_paths.append(entry.filename)
            if suffix in ANNOTATION_SUFFIXES:
                annotation_paths.append(entry.filename)

        return {
            "schema_version": "1.0",
            "archive_name": PurePosixPath(path).name,
            "archive_bytes": sum(entry.file_size for entry in entries),
            "member_count": len(entries),
            "image_member_count": len(image_paths),
            "annotation_member_count": len(annotation_paths),
            "suffix_counts": dict(sorted(suffixes.items())),
            "top_level_counts": dict(sorted(top_level.items())),
            "sample_image_members": image_paths[:10],
            "sample_annotation_members": annotation_paths[:10],
            "contains_source_images": False,
            "next_step": (
                "Map the released annotation format to a train/validation/test split "
                "only after confirming image-to-label joins from this inventory."
            ),
        }
