import numpy as np
from pupil_apriltags import Detector
# Parameters
TAG_SIZE_METERS = 0.04
HALF_SIZE = TAG_SIZE_METERS / 2.0
WARP_CORNER_IDS = [0, 1, 2, 3]
WARP_SCALE_PX_PER_M = 1000.0

# Map display configuration
MAP_SCALE_PX_PER_M = 1000.0  # 500 pixels per meter (1 cm = 5 px)
MARGIN_PX = 35               # Padding around field boundary

# Field demension
FIELD_WIDTH = 184 / 1000 # in meters
FIELD_LENGTH = 210 / 1000

OBJECT_POINTS = np.array([
    [-HALF_SIZE,  HALF_SIZE, 0.0],
    [ HALF_SIZE,  HALF_SIZE, 0.0],
    [ HALF_SIZE, -HALF_SIZE, 0.0],
    [-HALF_SIZE, -HALF_SIZE, 0.0]
], dtype=np.float32)

R_FLIP = np.array([
    [0,  1,  0],
    [1,  0,  0],
    [0,  0, -1]
], dtype=np.float32)

detector = Detector(
    families="tag36h11",
    nthreads=2,
    quad_decimate=1.0,
    refine_edges=1
)