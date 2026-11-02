"""Metric construction rules, calibrated only from R2R train."""

# Linear-interpolated terciles of all route-edge arc lengths in R2R train.
# 10,786 valid episodes; 33 episodes have fewer than two selected-level points.
DISTANCE_BIN_THRESHOLDS_M = (1.0000009599497723, 1.9999994364144549)
DISTANCE_BIN_CALIBRATION = "R2R/train: 10786 episodes, 53705 route edges"
MERGE_TRIGGER_DISTANCE_M = 1.0
NEAR_PATH_DISTANCE_M = 1.5
FORK_SNAP_RADIUS_M = 1.5
FORK_EDGE_MAX_M = 3.5
FORK_ANGLE_DEG = 60.0
