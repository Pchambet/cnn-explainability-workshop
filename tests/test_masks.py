import numpy as np
import pytest

from cnn_explainability.masks import CONDITIONS, LFW_LAYOUT, apply, apply_batch, black_box


@pytest.fixture
def image():
    rng = np.random.default_rng(0)
    return rng.integers(1, 255, size=(100, 80, 3), dtype=np.uint8)


def test_black_box_covers_exactly_the_fractional_box(image):
    out = black_box(image, (0.2, 0.4, 0.25, 0.75))
    assert (out[20:40, 20:60] == 0).all()
    untouched = np.ones(image.shape[:2], bool)
    untouched[20:40, 20:60] = False
    assert (out[untouched] == image[untouched]).all()


def test_input_is_never_modified(image):
    before = image.copy()
    for condition in CONDITIONS:
        apply(image, condition)
    assert (image == before).all()


def test_blur_changes_only_the_face_region_and_strong_blur_removes_more_detail(image):
    top, bottom, left, right = LFW_LAYOUT.face
    rows = slice(round(100 * top), round(100 * bottom))
    cols = slice(round(80 * left), round(80 * right))
    mild, strong = apply(image, "blur_mild"), apply(image, "blur_strong")
    outside = np.ones(image.shape[:2], bool)
    outside[rows, cols] = False
    assert (mild[outside] == image[outside]).all()
    detail = [np.diff(x[rows, cols].astype(float), axis=1).std() for x in (image, mild, strong)]
    assert detail[0] > detail[1] > detail[2]


def test_eye_bar_is_inside_the_eyes_and_nose_bar(image):
    eyes = apply(image, "eye_bar") == 0
    eyes_nose = apply(image, "eyes_nose_bar") == 0
    assert (eyes <= eyes_nose).all() and eyes_nose.sum() > eyes.sum()


def test_unknown_condition_raises(image):
    with pytest.raises(ValueError):
        apply(image, "sunglasses")


def test_batch_matches_single(image):
    batch = apply_batch(np.stack([image, image]), "eye_bar")
    assert (batch[1] == apply(image, "eye_bar")).all()
