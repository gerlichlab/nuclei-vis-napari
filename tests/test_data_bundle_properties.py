"""Verify various properties of nuclei data bundles."""

import os
from unittest import mock

import hypothesis as hyp
import hypothesis.extra.numpy as hyp_npy
import numpy as np
import pytest
from gertils.geometry import ImagePoint2D
from gertils.types import NucleusNumber
from hypothesis import strategies as st

from nuclei_vis_napari.data_bundles import (
    LOOPTRACE_NAPARI_NUCLEI_CHANNEL_ENV_VAR,
    NucleiVisualisationData,
    max_project_z,
)

gen_pixel_datatype = st.sampled_from((np.uint8, np.uint16))

gen_valid_image_2d = hyp_npy.arrays(
    dtype=gen_pixel_datatype,
    shape=st.tuples(
        st.integers(min_value=1, max_value=8),
        st.integers(min_value=1, max_value=8),
    ),
)

gen_image_5d__trivial_first_three = hyp_npy.arrays(
    dtype=gen_pixel_datatype,
    shape=st.tuples(
        st.just(1),
        st.just(1),
        st.just(1),
        st.integers(min_value=1, max_value=8),
        st.integers(min_value=1, max_value=8),
    ),
)

gen_valid_masks_2d = gen_valid_image_2d

gen_masks_5d__trivial_first_three = gen_image_5d__trivial_first_three

gen_nuc_num = st.integers(min_value=1).map(NucleusNumber)

gen_pos_float = st.floats(min_value=0).filter(lambda x: x != 0)

gen_img_pt_2d = st.tuples(gen_pos_float, gen_pos_float).map(
    lambda yx: ImagePoint2D(y=yx[0], x=yx[1])
)

gen_valid_centers = st.tuples(
    st.sets(gen_nuc_num).map(lambda ns: sorted(ns)), st.lists(gen_img_pt_2d)
).map(lambda nuclei_and_points: list(zip(*nuclei_and_points, strict=False)))


@st.composite
def gen_image_5d_invalid_because_of_nontrivial_first_dimension(draw):
    gen_dim = st.integers(min_value=1, max_value=8)
    img = draw(
        hyp_npy.arrays(
            dtype=gen_pixel_datatype, shape=st.tuples(gen_dim, gen_dim, gen_dim, gen_dim, gen_dim)
        )
    )
    hyp.assume(img.shape[0] != 1)
    return img


@hyp.given(
    bundle_expectation_pair=st.tuples(
        st.one_of(
            gen_image_5d__trivial_first_three.map(lambda img: (img, None)),
            gen_image_5d_invalid_because_of_nontrivial_first_dimension().map(
                lambda img: (
                    img,
                    ValueError("5D image for nuclei visualisation must have trivial first"),
                )
            ),
        ),
        st.one_of(gen_valid_masks_2d, gen_masks_5d__trivial_first_three),
        gen_valid_centers,
    ).map(lambda values: ((values[0][0], values[1], values[2]), values[0][1]))
)
@hyp.settings(phases=tuple(p for p in hyp.Phase if p != hyp.Phase.shrink))
def test_five_dimensional_image(bundle_expectation_pair):
    (image, masks, centers), expectation = bundle_expectation_pair
    assert len(image.shape) == 5, f"Generataed image that's not 5D, but {len(image.shape)}D"  # noqa: PLR2004
    match expectation:
        case None:
            observed = NucleiVisualisationData(image=image, masks=masks, centers=centers)
            expected = NucleiVisualisationData(image=image[0][0][0], masks=masks, centers=centers)
            assert observed == expected
        case ValueError() as e:
            with pytest.raises(ValueError, match=str(e)):
                NucleiVisualisationData(image=image, masks=masks, centers=centers)
        case other_event:
            raise RuntimeError(f"Unmatched expectation case: {other_event}")


@hyp.given(
    image=hyp_npy.arrays(
        dtype=gen_pixel_datatype,
        shape=st.lists(st.integers(min_value=1, max_value=10), min_size=3, max_size=3).map(
            lambda shape: (1, *tuple(shape))
        ),
    ),
    masks=st.one_of(gen_valid_masks_2d, gen_masks_5d__trivial_first_three),
    centers=gen_valid_centers,
)
def test_four_dimensional_image_with_trivial_channel_axis_matches_behavior_of_three_dimensional(
    image, masks, centers
):
    assert len(image.shape) == 4, f"Generated image isn't 4D, but {len(image.shape)}D"  # noqa: PLR2004
    expected = NucleiVisualisationData(image=image[0], masks=masks, centers=centers)
    observed = NucleiVisualisationData(image=image, masks=masks, centers=centers)
    assert observed == expected


@hyp.given(
    image=hyp_npy.arrays(
        dtype=gen_pixel_datatype,
        shape=st.lists(st.integers(min_value=1, max_value=8), min_size=4, max_size=4).filter(
            lambda shape: shape[0] > 1
        ),
    ),
    masks=st.one_of(gen_valid_masks_2d, gen_masks_5d__trivial_first_three),
    centers=gen_valid_centers,
    bad_nuc_channel=st.text(min_size=1),
)
@hyp.settings(phases=tuple(p for p in hyp.Phase if p != hyp.Phase.shrink))
def test_four_dimensional_image_with_nontrivial_channel_axis_and_non_int_nuc_channel(
    image, masks, centers, bad_nuc_channel
):
    assert len(image.shape) == 4, f"Generated image isn't 4D, but {len(image.shape)}D"  # noqa: PLR2004

    def is_int_like(s: str):
        """Check whether the given string is parsable as integer."""
        try:
            int(s)
        except Exception:  # noqa: BLE001
            return False
        return True

    # Ensure the generated value for nucleus channel can't be parsed as int, then set it.
    hyp.assume(not is_int_like(bad_nuc_channel))

    # Verify the property under test, that the appropriate error and message arise in this context.
    with (
        mock.patch(
            "nuclei_vis_napari.data_bundles.os.getenv",
            side_effect=lambda v: bad_nuc_channel
            if v == LOOPTRACE_NAPARI_NUCLEI_CHANNEL_ENV_VAR
            else os.getenv(v),
        ),
        pytest.raises(ValueError) as error_context,  # noqa: PT011
    ):
        NucleiVisualisationData(image=image, masks=masks, centers=centers)
    obs_msg = str(error_context.value)
    exp_msg = f"Illegal nuclei channel value (from {LOOPTRACE_NAPARI_NUCLEI_CHANNEL_ENV_VAR}): {bad_nuc_channel}"
    assert obs_msg == exp_msg


@hyp.given(
    image=hyp_npy.arrays(
        dtype=gen_pixel_datatype,
        shape=st.lists(st.integers(min_value=1, max_value=8), min_size=4, max_size=4).filter(
            lambda shape: shape[0] > 1
        ),
    ),
    masks=st.one_of(gen_valid_masks_2d, gen_masks_5d__trivial_first_three),
    centers=gen_valid_centers,
    bad_nuc_channel=st.integers(),
)
@hyp.settings(phases=tuple(p for p in hyp.Phase if p != hyp.Phase.shrink))
def test_four_dimensional_image_with_nontrivial_channel_axis_and_illegal_int_nuc_channel(
    image, masks, centers, bad_nuc_channel
):
    hyp.assume(bad_nuc_channel < 0 or bad_nuc_channel >= image.shape[0])
    assert len(image.shape) == 4, f"Generated image isn't 4D, but {len(image.shape)}D"  # noqa: PLR2004
    with (
        mock.patch(
            "nuclei_vis_napari.data_bundles.os.getenv",
            side_effect=lambda v: str(bad_nuc_channel)
            if v == LOOPTRACE_NAPARI_NUCLEI_CHANNEL_ENV_VAR
            else os.getenv(v),
        ),
        pytest.raises(ValueError) as error_context,  # noqa: PT011
    ):
        NucleiVisualisationData(image=image, masks=masks, centers=centers)
    obs_msg = str(error_context.value)
    exp_msg = f"Illegal nuclei channel value (from {LOOPTRACE_NAPARI_NUCLEI_CHANNEL_ENV_VAR}), {bad_nuc_channel}, for channel axis of length {image.shape[0]}"
    assert obs_msg == exp_msg


@hyp.given(
    image=hyp_npy.arrays(
        dtype=gen_pixel_datatype,
        shape=st.lists(st.integers(min_value=1, max_value=8), min_size=4, max_size=4).filter(
            lambda shape: shape[0] > 1
        ),
    ),
    masks=st.one_of(gen_valid_masks_2d, gen_masks_5d__trivial_first_three),
    centers=gen_valid_centers,
)
@hyp.settings(phases=tuple(p for p in hyp.Phase if p != hyp.Phase.shrink))
def test_four_dimensional_image_with_nontrivial_channel_axis_and_no_nuc_channel(
    image, masks, centers
):
    assert os.getenv(LOOPTRACE_NAPARI_NUCLEI_CHANNEL_ENV_VAR) is None
    with pytest.raises(ValueError) as error_context:  # noqa: PT011
        NucleiVisualisationData(image=image, masks=masks, centers=centers)
    exp_msg = f"When using nuclei images with multiple channels, nuclei channel must be specified through environment variable {LOOPTRACE_NAPARI_NUCLEI_CHANNEL_ENV_VAR}."
    assert str(error_context.value) == exp_msg


@st.composite
def gen_4d_image_and_nuc_channel(draw):
    image = draw(
        hyp_npy.arrays(
            dtype=gen_pixel_datatype,
            shape=st.lists(st.integers(min_value=1, max_value=8), min_size=4, max_size=4).filter(
                lambda shape: shape[0] > 1
            ),
        )
    )
    channel = draw(st.integers(min_value=0, max_value=image.shape[0] - 1))
    return image, channel


@hyp.given(
    image_and_channel=gen_4d_image_and_nuc_channel(),
    masks=st.one_of(gen_valid_masks_2d, gen_masks_5d__trivial_first_three),
    centers=gen_valid_centers,
)
@hyp.settings(phases=tuple(p for p in hyp.Phase if p != hyp.Phase.shrink))
def test_four_dimensional_image_with_nontrivial_channel_axis_and_but_good_nuc_channel(
    image_and_channel, masks, centers
):
    image, channel = image_and_channel
    with mock.patch(
        "nuclei_vis_napari.data_bundles.os.getenv",
        side_effect=lambda v: str(channel)
        if v == LOOPTRACE_NAPARI_NUCLEI_CHANNEL_ENV_VAR
        else os.getenv(v),
    ):
        observed = NucleiVisualisationData(image=image, masks=masks, centers=centers)
    expected = NucleiVisualisationData(image=image[channel], masks=masks, centers=centers)
    assert observed == expected


@hyp.given(
    image=hyp_npy.arrays(
        dtype=gen_pixel_datatype,
        shape=st.lists(st.integers(min_value=1, max_value=10), min_size=3, max_size=3),
    ),
    masks=st.one_of(gen_valid_masks_2d, gen_masks_5d__trivial_first_three),
    centers=gen_valid_centers,
)
def test_three_dimensional_image(image, masks, centers):
    assert len(image.shape) == 3, f"Generated image isn't 3D, but {len(image.shape)}D"  # noqa: PLR2004
    expected = NucleiVisualisationData(
        image=image[0] if image.shape[0] == 1 else np.max(image, axis=0),
        masks=masks,
        centers=centers,
    )
    observed = NucleiVisualisationData(image=image, masks=masks, centers=centers)
    assert observed == expected


gen_image_with_too_many_or_too_few_dimensions = st.one_of(
    hyp_npy.arrays(dtype=gen_pixel_datatype, shape=st.integers(min_value=1, max_value=100)),
    hyp_npy.arrays(
        dtype=gen_pixel_datatype,
        # Don't let too big an array be generated; 3^8 is already big
        shape=st.lists(st.integers(min_value=1, max_value=3), min_size=6, max_size=8),
    ),
)


@hyp.given(
    image=gen_image_with_too_many_or_too_few_dimensions,
    masks=st.one_of(gen_valid_masks_2d, gen_masks_5d__trivial_first_three),
    centers=gen_valid_centers,
)
def test_image_of_more_than_five_or_fewer_than_two_dimensions_is_illegal(image, masks, centers):
    with pytest.raises(ValueError) as error_context:  # noqa: PT011
        NucleiVisualisationData(image=image, masks=masks, centers=centers)
    assert (
        str(error_context.value)
        == f"Cannot use image with {len(image.shape)} dimension(s) for nuclei visualisation"
    )


@hyp.given(
    image=st.one_of(gen_valid_image_2d, gen_image_5d__trivial_first_three),
    centers=gen_valid_centers,
    masks_and_expectation=st.one_of(
        hyp_npy.arrays(
            dtype=gen_pixel_datatype,
            shape=st.lists(st.integers(min_value=1, max_value=3), min_size=1, max_size=8),
        )
        .filter(lambda masks: len(masks.shape) not in [2, 5])
        .map(
            lambda masks: (
                masks,
                ValueError(
                    f"Need 2D image for nuclear masks! Got {len(masks.shape)}: {masks.shape}"
                ),
            )
        ),
        st.one_of(gen_valid_masks_2d, gen_masks_5d__trivial_first_three).map(
            lambda masks: (masks, None)
        ),
    ),
)
def test_any_masks_must_be_2d_or_5d(image, centers, masks_and_expectation):
    masks, expected = masks_and_expectation
    match expected:
        case None:
            try:
                NucleiVisualisationData(image=image, masks=masks, centers=centers)
            except Exception as e:  # noqa: BLE001
                pytest.fail(f"Expected success but got error: {e}")
        case ValueError() as e:
            with pytest.raises(ValueError) as error_context:  # noqa: PT011
                NucleiVisualisationData(image=image, masks=masks, centers=centers)
            assert str(error_context.value) == str(e)
        case other_event:
            raise RuntimeError(f"Unmatched expectation case: {other_event}")


@hyp.given(
    image=st.one_of(gen_valid_image_2d, gen_image_5d__trivial_first_three),
    masks=hyp_npy.arrays(
        dtype=gen_pixel_datatype,
        shape=st.lists(st.integers(min_value=1, max_value=5), min_size=5, max_size=5),
    ),
    centers=gen_valid_centers,
)
def test_5d_masks_without_trivial_initial_dimensions_gives_expected_error(image, masks, centers):
    assert len(masks.shape) == 5, f"Generated {len(image.shape)}D masks, not 5D"  # noqa: PLR2004
    hyp.assume(any(d > 1 for d in masks.shape[:3]))
    with pytest.raises(ValueError) as error_context:  # noqa: PT011
        NucleiVisualisationData(image=image, masks=masks, centers=centers)
    obs_msg = str(error_context.value)
    exp_msg = f"5D nuclear masks image with at least 1 nontrivial (t, c, z) axis! {masks.shape}"
    assert obs_msg == exp_msg


@hyp.given(
    image=hyp_npy.arrays(
        dtype=gen_pixel_datatype, shape=st.lists(st.integers(min_value=1, max_value=5), max_size=5)
    )
)
def test_max_z_projection_fails_when_image_is_not_3d(image):
    hyp.assume(len(image.shape) != 3)  # noqa: PLR2004
    with pytest.raises(ValueError) as error_context:  # noqa: PT011
        max_project_z(image)
    obs_msg = str(error_context.value)
    exp_msg = f"Image to max-z-project must have 3 dimensions! Got {len(image.shape)}"
    assert obs_msg == exp_msg
