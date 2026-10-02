import pytest

from cnn_explainability.pipeline import summarise


def test_summarise_groups_by_representation_gallery_and_condition():
    rows = [
        {"representation": "a", "gallery": "clean", "condition": "none", "draw": d, "rank1": v}
        for d, v in enumerate([0.1, 0.3])
    ] + [
        {
            "representation": "a",
            "gallery": "matched",
            "condition": "none",
            "draw": 0,
            "rank1": 0.9,
            "extra": 1.0,
        }
    ]
    out = summarise(rows)
    assert len(out) == 2
    clean = next(r for r in out if r["gallery"] == "clean")
    assert clean["rank1_mean"] == pytest.approx(0.2)
    assert clean["rank1_lo"] == pytest.approx(0.105)
    assert "extra_mean" not in clean
