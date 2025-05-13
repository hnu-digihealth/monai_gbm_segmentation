"""
Preprocessing module for image normalization.

Contains MONAI-compatible transform to apply H&E normalization
to images during training, validation, and testing.
"""

# Python Standard Libraries
import logging
from typing import Any, Sequence

# Third Party Libraries
from monai.transforms import MapTransform

logger = logging.getLogger("Preprocessing")


logger.info("Preprocessing module loaded")


class HENormalization(MapTransform):
    """
    Apply H&E stain normalization using a given normalizer object.

    This class wraps a stain normalizer to be used as a MONAI MapTransform.
    Supported methods are "reinhard" and other callable normalizers
    that return the normalized image (and optionally other outputs).

    Args:
        keys (list[str]): Keys of the data dictionary to apply the transform on.
        normalizer (object): An initialized stain normalizer with a `.normalize(I=...)` method.
        method (str): Normalization method (e.g., "reinhard", "vahadane", etc.).

    Returns:
        dict: Transformed data dictionary with normalized image tensors.
    """

    def __init__(self, keys: Sequence[str], normalizer: Any, method: str) -> None:
        super().__init__(keys)
        self.normalizer = normalizer
        self.method = method

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        d = dict(data)
        for key in self.keys:
            if self.method == "reinhard":
                img = self.normalizer.normalize(I=d[key])
            else:
                img, _, _ = self.normalizer.normalize(I=d[key])
            d[key] = img
        return d
