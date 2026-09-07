from zipfile import ZipFile

from aerosynth_eval.dlr_dent_detection import audit_capture_leakage, scan_archive, session_groups


def _write(archive: ZipFile, split: str, image_id: str, label: str | None) -> None:
    root = f"release/{split}"
    archive.writestr(f"{root}/images/{image_id}-image.png", b"not-read-by-scan")
    archive.writestr(f"{root}/masks/{image_id}-masks.png", b"mask")
    if label is not None:
        archive.writestr(f"{root}/labels/{image_id}-image.txt", label)


def test_scans_yolo_labels_and_retains_unlabelled_images(tmp_path) -> None:
    path = tmp_path / "dlr.zip"
    with ZipFile(path, "w") as archive:
        _write(archive, "train", "1000", "0 0.5 0.5 0.2 0.2\n")
        _write(archive, "val", "1010", None)
    records = scan_archive(str(path))
    assert [record.has_dent for record in records] == [True, False]
    assert records[0].boxes[0] == (0, 0.5, 0.5, 0.2, 0.2)


def test_session_audit_detects_cross_release_split_capture_groups(tmp_path) -> None:
    path = tmp_path / "dlr.zip"
    with ZipFile(path, "w") as archive:
        _write(archive, "train", "1000", "")
        _write(archive, "test", "1050", "")
        _write(archive, "test", "2000", "")
    records = scan_archive(str(path))
    audit = audit_capture_leakage(records, gap_seconds=60)
    assert audit["capture_sessions"] == 2
    assert audit["cross_source_split_sessions"] == 1
    assert len(set(session_groups(records, 60).values())) == 2
