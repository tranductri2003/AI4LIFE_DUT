from IPython import display
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from ultralytics import YOLO
import tkinter
import supervision as sv
MODEL = "yolov8s.pt"
model = YOLO(MODEL)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

SAMPLE_VIDEO_PATH = "video.mp4"
# print(sv.VideoInfo.from_video_path(SAMPLE_VIDEO_PATH))
# VideoInfo(width=1920, height=1080, fps=21, total_frames=2921)


def process_frame(frame: np.ndarray, _) -> np.ndarray:
    # detect
    results = model(frame, imgsz=1280)[0]
    detections = sv.Detections.from_yolov8(results)
    detections = detections[detections.class_id == 0]
    zone.trigger(detections=detections)

    # annotate
    box_annotator = sv.BoxAnnotator(
        thickness=4, text_thickness=4, text_scale=2)
    labels = [f"{model.names[class_id]} {confidence:0.2f}" for _,
              confidence, class_id, _ in detections]
    frame = box_annotator.annotate(
        scene=frame, detections=detections, labels=labels)
    frame = zone_annotator.annotate(scene=frame)

    return frame


# initiate polygon zone
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

# initiate annotators
box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
zone_annotator = sv.PolygonZoneAnnotator(
    zone=zone, color=sv.Color.white(), thickness=6, text_thickness=6, text_scale=4)

# extract video frame
generator = sv.get_video_frames_generator(SAMPLE_VIDEO_PATH)
iterator = iter(generator)
frame = next(iterator)


sv.process_video(source_path=SAMPLE_VIDEO_PATH,
                 target_path=f"result.mp4", callback=process_frame)
display.clear_output()
# x = np.linspace(0, 20, 100)
# mpl.use('TkAgg')
# plt.show()
# sv.show_frame_in_notebook(frame, (16, 16))
