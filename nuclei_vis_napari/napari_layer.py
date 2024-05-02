"""Abstractions for working with Napari layers"""

import dataclasses
from enum import Enum
from typing import Literal

import numpy.typing as npt
from numpydoc_decorator import doc  # type: ignore[import-untyped]

LayerData = npt.ArrayLike
LayerParams = dict[str, object]


class NapariLayerType(Enum):
    """The layer types allowed by Napari"""

    Image = "image"
    Labels = "labels"
    Points = "points"
    Shapes = "shapes"
    Surface = "surface"
    Tracks = "tracks"
    Vectors = "vectors"


@doc(
    summary="More name-oriented and typed bundle of data for a Napari layer",
    parameters=dict(
        data="The actual data to visualise for the Napari layer",
        parameters="The parameters to pass to the layer constructor",
        get_type="Representation of one of the few valid Napari layer types",
    ),
)
@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class NapariLayer:  # noqa: D101
    data: LayerData
    parameters: LayerParams
    get_type: NapariLayerType

    def __eq__(self, value: object) -> bool:
        raise NotImplementedError("Do not directly compare Napari layers!")

    def _is_of_type(self, t: NapariLayerType) -> bool:
        return self.get_type is t

    @property
    def is_image(self) -> bool:
        """Determine if this layer is an image layer."""
        return self._is_of_type(NapariLayerType.Image)

    @property
    def is_labels(self) -> bool:
        """Determine if this layer is a labels layer."""
        return self._is_of_type(NapariLayerType.Labels)

    @property
    def is_points(self) -> bool:
        """Determine if this layer is a points layer."""
        return self._is_of_type(NapariLayerType.Points)

    @property
    def as_image(self) -> tuple[LayerData, LayerParams, Literal["image"]]:  # type: ignore[return]
        """Cast this layer as an image layer, flattening to tuple."""
        if self.is_image:
            return self.data, self.parameters, "image"
        self._raise_cast_error(NapariLayerType.Image)  # noqa: RET503

    @property
    def as_labels(self) -> tuple[LayerData, LayerParams, Literal["labels"]]:  # type: ignore[return]
        """Cast this layer as a labels layer, flattening to tuple."""
        if self.is_labels:
            return self.data, self.parameters, "labels"
        self._raise_cast_error(NapariLayerType.Labels)  # noqa: RET503

    @property
    def as_points(self) -> tuple[LayerData, LayerParams, Literal["points"]]:  # type: ignore[return]
        """Cast this layer as a points layer, flattening to tuple."""
        if self.is_points:
            return self.data, self.parameters, "points"
        self._raise_cast_error(NapariLayerType.Points)  # noqa: RET503

    def _raise_cast_error(self, target_type: NapariLayerType) -> None:
        raise NotImplementedError(
            f"Cannot cast {self.get_type.value} layer as {target_type.value} layer"
        )
