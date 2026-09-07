"""Provenance checks for DLR's public aircraft-dent dataset release."""

from __future__ import annotations

import hashlib
from typing import Any

RECORD_ID = "17900121"
EXPECTED_DOI = "10.5281/zenodo.17900121"
EXPECTED_LICENSE = "mit-license"
EXPECTED_FILE = "plane-10.27-fulldata-split.zip"
MINIMUM_IMAGES = 6_000


def audit_record(record: dict[str, Any]) -> dict[str, Any]:
    """Validate the public Zenodo metadata without downloading the 5.7 GB archive."""

    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Zenodo record is missing metadata")
    if record.get("doi") != EXPECTED_DOI:
        raise ValueError(f"Unexpected DOI: {record.get('doi')!r}")
    license_data = metadata.get("license")
    if not isinstance(license_data, dict) or license_data.get("id") != EXPECTED_LICENSE:
        raise ValueError("The DLR aircraft-dent release is not marked MIT")
    files = record.get("files")
    if not isinstance(files, list):
        raise ValueError("Zenodo record is missing files")
    archive = next((item for item in files if item.get("key") == EXPECTED_FILE), None)
    if not isinstance(archive, dict):
        raise ValueError(f"Expected archive {EXPECTED_FILE!r} is unavailable")
    link = archive.get("links", {}).get("self")
    if not isinstance(link, str) or not link.startswith("https://zenodo.org/"):
        raise ValueError("The archive does not have an official Zenodo download URL")
    description = str(metadata.get("description", ""))
    mentions_image_count = "6000" in description.replace(",", "")
    if not mentions_image_count:
        raise ValueError("The release description does not state the expected image scale")
    return {
        "schema_version": "1.0",
        "dataset_id": "DLR aircraft dent detection",
        "doi": EXPECTED_DOI,
        "record_id": RECORD_ID,
        "license": "MIT",
        "release_file": {
            "name": EXPECTED_FILE,
            "bytes": int(archive["size"]),
            "download_url": link,
        },
        "reported_labelled_images": {"minimum": MINIMUM_IMAGES, "source": "release description"},
        "annotation_method": "optical-tracking-derived dent annotations",
        "acquisition_status": "metadata_verified_archive_not_downloaded",
        "contains_source_images": False,
        "limitations": [
            "The labels are optical-tracking-derived, not a new human review of generated images.",
            "The archive layout and final train/test split must be audited after download.",
            "A metadata receipt is not a downstream-utility result.",
        ],
    }


def source_digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
