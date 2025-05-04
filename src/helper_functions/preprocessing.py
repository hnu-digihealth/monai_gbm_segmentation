# Python Standard Libraries
import logging
from typing import Any, Sequence

# Third Party Libraries
from monai.transforms import MapTransform

logger = logging.getLogger("Preprocessing")


logger.info("Preprocessing module loaded")


class HENormalization(MapTransform):
    """Apply H&E normalization to images."""

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
