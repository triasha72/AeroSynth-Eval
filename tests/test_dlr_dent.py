from aerosynth_eval.dlr_dent import EXPECTED_DOI, EXPECTED_FILE, audit_record


def test_audits_the_official_dlr_release_metadata() -> None:
    result = audit_record(
        {
            "doi": EXPECTED_DOI,
            "metadata": {
                "license": {"id": "mit-license"},
                "description": "The release contains over 6000 labelled images.",
            },
            "files": [
                {
                    "key": EXPECTED_FILE,
                    "size": 5_740_757_344,
                    "links": {"self": "https://zenodo.org/api/files/archive/content"},
                }
            ],
        }
    )
    assert result["license"] == "MIT"
    assert result["reported_labelled_images"]["minimum"] == 6_000


def test_rejects_a_non_mit_release() -> None:
    record = {
        "doi": EXPECTED_DOI,
        "metadata": {"license": {"id": "cc-by-4.0"}, "description": "6000 images"},
        "files": [],
    }
    try:
        audit_record(record)
    except ValueError as error:
        assert "MIT" in str(error)
    else:
        raise AssertionError("A non-MIT release must be rejected")
