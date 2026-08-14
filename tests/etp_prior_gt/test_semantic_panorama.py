import unittest

import numpy as np

from prior.constants import MAPPED_OBJECT_NAMES, OBJECT_MAPPING, REGION_MAPPING
from vlnce_baselines.models.etp_prior_gt.semantic_panorama import (
    SENSOR_YAWS_DEGREES,
    panorama_semantic_label,
)


class _Category:
    def __init__(self, default_index, mpcat40_index=None):
        self.default_index = default_index
        self.mpcat40_index = mpcat40_index

    def index(self, mapping=None):
        if mapping == "mpcat40":
            return self.mpcat40_index
        return self.default_index


class _Region:
    category = _Category(0)


class _Object:
    id = "object_7"
    category = _Category(0, 1)
    region = _Region()


class _Scene:
    objects = [_Object()]


class SemanticPanoramaTest(unittest.TestCase):
    def test_visible_instance_sets_object_and_region_labels(self):
        observations = {}
        for yaw in SENSOR_YAWS_DEGREES:
            observations[f"depth_{yaw:03d}"] = np.ones((2, 2), dtype=np.float32)
            observations[f"semantic_{yaw:03d}"] = np.full(
                (2, 2), 7, dtype=np.int32
            )
        label = panorama_semantic_label(observations, _Scene())
        object_index = OBJECT_MAPPING[1]
        region_index = len(MAPPED_OBJECT_NAMES) + REGION_MAPPING[0]
        if object_index > 0:
            self.assertEqual(label[object_index], 1.0)
        self.assertEqual(label[region_index], 1.0)


if __name__ == "__main__":
    unittest.main()
