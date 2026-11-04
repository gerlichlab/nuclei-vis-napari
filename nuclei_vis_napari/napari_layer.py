"""Abstractions for working with Napari layers"""

import dataclasses
from enum import Enum
from typing import Literal

import numpy.typing as npt

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


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class NapariLayer:  # noqa: PLW1641
    """More name-oriented and typed bundle of data for a Napari layer.

    Parameters
    ----------
    data : array_like
        The actual data to visualise for the Napari layer.
    parameters : dict[str, object]
        The parameters to pass to the layer constructor.
    get_type : NapariLayerType
        Representation of one of the few valid Napari layer types.

    """

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
    def as_image(self) -> tuple[LayerData, LayerParams, Literal["image"]]:
        """Cast this layer as an image layer, flattening to tuple."""
        if self.is_image:
            return self.data, self.parameters, "image"
        raise self._cast_error(NapariLayerType.Image)

    @property
    def as_labels(self) -> tuple[LayerData, LayerParams, Literal["labels"]]:
        """Cast this layer as a labels layer, flattening to tuple."""
        if self.is_labels:
            return self.data, self.parameters, "labels"
        raise self._cast_error(NapariLayerType.Labels)

    @property
    def as_points(self) -> tuple[LayerData, LayerParams, Literal["points"]]:
        """Cast this layer as a points layer, flattening to tuple."""
        if self.is_points:
            return self.data, self.parameters, "points"
        raise self._cast_error(NapariLayerType.Points)

    def _cast_error(self, target_type: NapariLayerType) -> NotImplementedError:
        return NotImplementedError(
            f"Cannot cast {self.get_type.value} layer as {target_type.value} layer"
        )
