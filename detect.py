from IPython import display
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from ultralytics import YOLO
import tkinter
import supervision as sv
import cv2
from supervision import PolygonZone

MODEL = "yolov8n.pt"
model = YOLO(MODEL)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

SAMPLE_VIDEO_PATH = "video.mp4"
VIDEO_INFO = sv.VideoInfo.from_video_path(SAMPLE_VIDEO_PATH)

WIDTH = VIDEO_INFO.width
HEIGHT = VIDEO_INFO.height
FPS = VIDEO_INFO.fps
TOTAL_FRAMES = VIDEO_INFO.total_frames


def process_frame(frame: np.ndarray, _) -> np.ndarray:
    # 3 detect
    results = model(frame, imgsz=1280)[0]
    detections = sv.Detections.from_yolov8(results)
    detections = detections[(detections.class_id == 0)
                            & (detections.confidence > 0.4)]
    zone.trigger(detections=detections)

    # 4 annotate
    box_annotator = sv.BoxAnnotator(
        thickness=4, text_thickness=4, text_scale=2)
    labels = [f"{model.names[class_id]} {confidence:0.2f}" for _,
              confidence, class_id, _ in detections]
    frame = box_annotator.annotate(
        scene=frame, detections=detections, labels=labels)
    frame = zone_annotator.annotate(scene=frame)

    return frame


# 1 initiate polygon zone
polygon = np.array([
    [961, 1078],
    [763, 550],
    [1502, 254],
    [1914, 260],
    [1920, 1080]
])
video_info = sv.VideoInfo.from_video_path(SAMPLE_VIDEO_PATH)
zone = sv.PolygonZone(
    polygon=polygon, frame_resolution_wh=video_info.resolution_wh)

# 2 initiate annotators
box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
zone_annotator = sv.PolygonZoneAnnotator(
    zone=zone, color=sv.Color.white(), thickness=6, text_thickness=6, text_scale=4)


# Xuat ket qua ra file
# sv.process_video(source_path=SAMPLE_VIDEO_PATH,
#                  target_path=f"result_yolov8n.mp4", callback=process_frame)
# display.clear_output()


# Xuat ra man hinh

# extract video frame
generator = sv.get_video_frames_generator(SAMPLE_VIDEO_PATH)
iterator = iter(generator)
frame = next(iterator)

# matrix_rotate = np.array([[0.9397, 0.342], [-0.342, 0.9397]])
for frame in iterator:
    # frame = np.dot(frame, matrix_rotate)
    current = process_frame(frame, 1)

    width = int(current.shape[1] * 0.5)
    height = int(current.shape[0] * 0.5)
    new_size = (width, height)

    resized_img = cv2.resize(current, new_size)
    cv2.imshow('Resized Image', resized_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
