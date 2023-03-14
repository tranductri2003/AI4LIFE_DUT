from supervision.notebook.utils import show_frame_in_notebook
from supervision.video import get_video_frames_generator
from supervision.draw.color import ColorPalette
from supervision.detection.core import Detections, BoxAnnotator
from typing import List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import Callable, Generator, Optional, Tuple
from dataclasses import dataclass
from ultralytics import YOLO
import os
import scipy


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
MODEL = "yolov8n.pt"
model = YOLO(MODEL)
model.fuse()
VIDEO = "video.mp4"
model = YOLO("yolov8n.pt")


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []

    # transform matrix to biadjacency_matrix
    biadjacency_matrix = scipy.sparse.coo_matrix(cost_matrix)

    # Use scipy.sparse.csgraph.min_weight_full_bipartite_matching() from SciPy instead of lap.lapjv
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.min_weight_full_bipartite_matching.html
    x, y = scipy.sparse.csgraph.min_weight_full_bipartite_matching(
        biadjacency_matrix, maximize=True)

    matches.extend([ix, mx] for ix, mx in enumerate(x) if mx >= 0)
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)

    return matches, unmatched_a, unmatched_b


results = model.track(source="video.mp4",
                      stream=True,
                      show=True,
                      tracker="bytetrack.yaml")
