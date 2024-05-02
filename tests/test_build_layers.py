"""Tests for specific behaviors and properties of layer construction"""

from unittest import mock

import hypothesis as hyp
import pytest
from gertils.types import FieldOfViewFrom1
from hypothesis import strategies as st

from nuclei_vis_napari.reader import build_layers


@hyp.given(
    shape1=st.lists(st.integers(min_value=1), min_size=2, max_size=5).map(tuple),
    shape2=st.lists(st.integers(min_value=1), min_size=2, max_size=5).map(tuple),
)
@hyp.settings(phases=tuple(p for p in hyp.Phase if p != hyp.Phase.shrink))
def test_images_of_different_sizes_raises_expected_error(shape1, shape2):
    hyp.assume(shape1 != shape2)
    bundles = {
        FieldOfViewFrom1(1): mock.MagicMock(
            image=mock.MagicMock(shape=shape1), masks=mock.MagicMock(shape=shape1)
        ),
        FieldOfViewFrom1(2): mock.MagicMock(
            image=mock.MagicMock(shape=shape2), masks=mock.MagicMock(shape=shape1)
        ),
    }
    # 0-based counting when messaging
    exp_msg = f"Image shape for FOV 1 doesn't match previous: {shape2} != {shape1}"
    with pytest.raises(RuntimeError) as error_context:
        build_layers(bundles)
    obs_msg = str(error_context.value)
    assert obs_msg == exp_msg


@hyp.given(
    shape1=st.lists(st.integers(min_value=1), min_size=2, max_size=5).map(tuple),
    shape2=st.lists(st.integers(min_value=1), min_size=2, max_size=5).map(tuple),
)
@hyp.settings(phases=tuple(p for p in hyp.Phase if p != hyp.Phase.shrink))
def test_masks_of_different_sizes_raises_expected_error(shape1, shape2):
    hyp.assume(shape1 != shape2)
    bundles = {
        FieldOfViewFrom1(1): mock.MagicMock(
            image=mock.MagicMock(shape=shape1), masks=mock.MagicMock(shape=shape1)
        ),
        FieldOfViewFrom1(2): mock.MagicMock(
            image=mock.MagicMock(shape=shape1), masks=mock.MagicMock(shape=shape2)
        ),
    }
    # 0-based counting when messaging
    exp_msg = f"Masks shape for FOV 1 doesn't match previous: {shape2} != {shape1}"
    with pytest.raises(RuntimeError) as error_context:
        build_layers(bundles)
    obs_msg = str(error_context.value)
    assert obs_msg == exp_msg


@hyp.given(
    shape1=st.lists(st.integers(min_value=1), min_size=2, max_size=5).map(tuple),
    shape2=st.lists(st.integers(min_value=1), min_size=2, max_size=5).map(tuple),
)
@hyp.settings(phases=tuple(p for p in hyp.Phase if p != hyp.Phase.shrink))
def test_mismatch_between_image_shape_and_masks_shape_raises_expected_error(shape1, shape2):
    hyp.assume(shape1 != shape2)
    bundles = {
        FieldOfViewFrom1(1): mock.MagicMock(
            image=mock.MagicMock(shape=shape1), masks=mock.MagicMock(shape=shape2)
        ),
    }
    # 0-based counting when messaging
    exp_msg = f"Masks shape for FOV 0 doesn't match previous: {shape2} != {shape1}"
    with pytest.raises(RuntimeError) as error_context:
        build_layers(bundles)
    obs_msg = str(error_context.value)
    assert obs_msg == exp_msg
