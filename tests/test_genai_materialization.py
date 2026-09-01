from scripts.materialize_genai_bench import image_suffix


def test_image_suffix_preserves_known_formats_and_rejects_unknown_ones() -> None:
    assert image_suffix({"path": "sample.webp"}) == ".webp"
    assert image_suffix({"path": "sample.bmp"}) == ".png"
    assert image_suffix({"path": None}) == ".png"
