import unittest

import numpy as np

from vlnce_baselines.models.etp_prior_gt.panorama_semantic_cache import (
    _agent_pose_from_connectivity,
)


class PanoramaSemanticCacheTest(unittest.TestCase):
    def test_connectivity_pose_becomes_habitat_compatible_arrays(self):
        pose = [
            1,
            0,
            0,
            9,
            0,
            1,
            0,
            2.76605,
            0,
            0,
            1,
            1.62137,
            0,
            0,
            0,
            1,
        ]

        position, rotation = _agent_pose_from_connectivity(pose)

        np.testing.assert_array_equal(
            position,
            np.asarray((9.0, 2.76605, 1.62137), dtype=np.float32),
        )
        np.testing.assert_array_equal(
            rotation,
            np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32),
        )
        self.assertEqual(position.dtype, np.float32)
        self.assertEqual(rotation.dtype, np.float32)

    def test_short_pose_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least 12 values"):
            _agent_pose_from_connectivity([0.0] * 11)

    def test_non_finite_position_is_rejected(self):
        pose = [0.0] * 16
        pose[7] = float("nan")

        with self.assertRaisesRegex(ValueError, "non-finite position"):
            _agent_pose_from_connectivity(pose)


if __name__ == "__main__":
    unittest.main()
