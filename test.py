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
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
MODEL = "yolov8n.pt"
model = YOLO(MODEL)
model.fuse()
VIDEO = "video.mp4"


def show_frame_in_notebook(frame: np.ndarray, size: Tuple[int, int] = (10, 10), cmap: str = "gray"):
    if frame.ndim == 2:
        plt.figure(figsize=size)
        plt.imshow(frame, cmap=cmap)
    else:
        plt.figure(figsize=size)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return frame


generator = get_video_frames_generator(VIDEO)

box_annotator = BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)

iterator = iter(generator)


def resize_frame(frame, scale_percent):
    width = int(frame.shape[1]*scale_percent)
    height = int(frame.shape[0]*scale_percent)
    dim = (width, height)
    frame = cv2.resize(frame, dim)
    return frame


for frame in iterator:
    frame = resize_frame(frame, 0.6)

    results = model(frame)[0]
    detections = Detections(
        xyxy=results.boxes.xyxy.cpu().numpy(),
        confidence=results.boxes.conf.cpu().numpy(),
        class_id=results.boxes.cls.cpu().numpy().astype(int)
    )
    frame = box_annotator.annotate(scene=frame, detections=detections)
    # frame = show_frame_in_notebook(frame, (640, 640))

    cv2.imshow("Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

