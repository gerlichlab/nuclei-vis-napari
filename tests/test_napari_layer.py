"""Tests for general Napari layer properties"""

from typing import Optional

import hypothesis as hyp
import hypothesis.extra.numpy as hyp_npy
import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import strategies as st

from nuclei_vis_napari.napari_layer import NapariLayer, NapariLayerType

gen_any_array = hyp_npy.arrays(
    dtype=st.sampled_from((bool, float, int, str)),
    # Keep to relatively manageable size.
    shape=hyp_npy.array_shapes(max_dims=5, max_side=5),
)


@st.composite
def gen_napari_layer(draw, layer_type: Optional[NapariLayerType] = None):
    data = draw(gen_any_array)
    params = draw(
        st.dictionaries(
            # Keep to relatively manageable size.
            keys=st.one_of(st.characters(), st.integers(), st.text(max_size=10)),
            values=st.one_of(st.booleans(), st.characters(), st.floats(), st.integers(), st.text()),
            # Keep to relatively manageable size.
            max_size=4,
        )
    )
    layer_type = layer_type or draw(st.from_type(NapariLayerType))
    return NapariLayer(data=data, parameters=params, get_type=layer_type)


@hyp.given(
    layer1=gen_napari_layer(),
    layer2=gen_napari_layer(),
)
@hyp.settings(phases=tuple(p for p in hyp.Phase if p != hyp.Phase.shrink))
def test_equivalence_test_is_prohibited_for_napari_layers(layer1, layer2):
    assert isinstance(layer1, NapariLayer)
    assert isinstance(layer2, NapariLayer)
    with pytest.raises(NotImplementedError, match="Do not directly compare Napari layers!"):
        assert layer1 == layer2


@hyp.given(
    layer_and_expectation=st.one_of(
        gen_napari_layer(layer_type=NapariLayerType.Image).map(lambda layer: (layer, True)),
        gen_napari_layer()
        .filter(lambda layer: layer.get_type != NapariLayerType.Image)
        .map(lambda layer: (layer, False)),
    )
)
def test_image_check_is_correct(layer_and_expectation):
    layer, expectation = layer_and_expectation
    assert layer.is_image is expectation


@hyp.given(
    layer_and_expectation=st.one_of(
        gen_napari_layer(layer_type=NapariLayerType.Labels).map(lambda layer: (layer, True)),
        gen_napari_layer()
        .filter(lambda layer: layer.get_type != NapariLayerType.Labels)
        .map(lambda layer: (layer, False)),
    )
)
def test_labels_check_is_correct(layer_and_expectation):
    layer, expectation = layer_and_expectation
    assert layer.is_labels is expectation


@hyp.given(
    layer_and_expectation=st.one_of(
        gen_napari_layer(layer_type=NapariLayerType.Points).map(lambda layer: (layer, True)),
        gen_napari_layer()
        .filter(lambda layer: layer.get_type != NapariLayerType.Points)
        .map(lambda layer: (layer, False)),
    )
)
def test_points_check_is_correct(layer_and_expectation):
    layer, expectation = layer_and_expectation
    assert layer.is_points is expectation


@hyp.given(
    layer=st.one_of(
        gen_napari_layer(layer_type=NapariLayerType.Image),
        gen_napari_layer().filter(lambda layer: not layer.is_image),
    )
)
def test_layer_cast_to_image_layer_is_correct(layer):
    if layer.is_image:
        data, params, name = layer.as_image
        assert smart_array_equal(data, layer.data)
        assert params == layer.parameters
        assert name == "image"
    else:
        with pytest.raises(
            NotImplementedError, match=f"Cannot cast {layer.get_type.value} layer as image layer"
        ):
            layer.as_image  # noqa: B018


@hyp.given(
    layer=st.one_of(
        gen_napari_layer(layer_type=NapariLayerType.Labels),
        gen_napari_layer().filter(lambda layer: not layer.is_labels),
    )
)
def test_layer_cast_to_labels_layer_is_correct(layer):
    if layer.is_labels:
        data, params, name = layer.as_labels
        assert smart_array_equal(data, layer.data)
        assert params == layer.parameters
        assert name == "labels"
    else:
        with pytest.raises(
            NotImplementedError, match=f"Cannot cast {layer.get_type.value} layer as labels layer"
        ):
            layer.as_labels  # noqa: B018


@hyp.given(
    layer=st.one_of(
        gen_napari_layer(layer_type=NapariLayerType.Points),
        gen_napari_layer().filter(lambda layer: not layer.is_points),
    )
)
def test_layer_cast_to_points_layer_is_correct(layer):
    if layer.is_points:
        data, params, name = layer.as_points
        assert smart_array_equal(data, layer.data)
        assert params == layer.parameters
        assert name == "points"
    else:
        with pytest.raises(
            NotImplementedError, match=f"Cannot cast {layer.get_type.value} layer as points layer"
        ):
            layer.as_points  # noqa: B018


def smart_array_equal(a1: npt.ArrayLike, a2: npt.ArrayLike):
    if np.array_equal(a1, a2):
        return True
    try:
        return np.array_equal(a1, a2, equal_nan=True)
    except TypeError:  # non-numeric data type, probably
        return False
