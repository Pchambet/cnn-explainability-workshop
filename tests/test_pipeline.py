import numpy as np
import pytest

from cnn_explainability.pipeline import bar_scan, region_shares, summarise


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


def test_region_shares_clip_negative_sensitivity_and_report_area():
    heat = np.zeros((10, 10))
    heat[0:5, 0:5] = 1.0  # all positive mass in the top-left quarter
    heat[5:, 5:] = -3.0  # a rise in similarity is not sensitivity
    shares = region_shares(heat, {"quarter": (0.0, 0.5, 0.0, 0.5), "half": (0.0, 1.0, 0.5, 1.0)})
    assert shares["quarter"] == pytest.approx({"sensitivity_share": 1.0, "area_share": 0.25})
    assert shares["half"]["sensitivity_share"] == pytest.approx(0.0)


def test_bar_scan_finds_the_rows_where_a_same_size_bar_hides_most_and_least():
    heat = np.ones((20, 10))
    heat[12:14] = 5.0  # a hot two-row band
    heat[2:4] = 0.0  # a cold two-row band
    scan = bar_scan(heat, bar=(0.0, 0.1, 0.0, 1.0), rows=(0.0, 1.0))
    assert scan["max"]["top"] == pytest.approx(0.6)
    assert scan["max"]["sensitivity_share"] == pytest.approx(10 / 26)
    assert scan["min"]["top"] == pytest.approx(0.1)
    assert scan["min"]["sensitivity_share"] == pytest.approx(0.0)
