from zipfile import ZipFile

from aerosynth_eval.dlr_dent_archive import inspect_archive


def test_inspect_archive_counts_members_without_extracting(tmp_path) -> None:
    archive = tmp_path / "dlr.zip"
    with ZipFile(archive, "w") as output:
        output.writestr("images/dent-1.jpg", b"not-an-image")
        output.writestr("labels/dent-1.json", b"{}")
        output.writestr("README.md", b"metadata")

    receipt = inspect_archive(str(archive))

    assert receipt["member_count"] == 3
    assert receipt["image_member_count"] == 1
    assert receipt["annotation_member_count"] == 1
    assert receipt["contains_source_images"] is False
