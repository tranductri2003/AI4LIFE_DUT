import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from ultralytics import YOLO
import supervision as sv
import cv2
from supervision import PolygonZone
from IPython import display


MODEL = "yolov8n.pt"
model = YOLO(MODEL)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

SOURCE_VIDEO_PATH = "video1606_10.mp4"
TARGET_VIDEO_PATH = "result_video1606_10.mp4"
VIDEO_INFO = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)


WIDTH = VIDEO_INFO.width
HEIGHT = VIDEO_INFO.height
FPS = VIDEO_INFO.fps
TOTAL_FRAMES = VIDEO_INFO.total_frames


personPerFrame = [0, 0]
res = 0
N = 20
personPerNFrame = [0, 0]
i = 0
sumPerson = 0


def process_frame(frame: np.ndarray, _,res) -> np.ndarray:
    # detect
    results = model(frame, imgsz=1280)[0]
    detections = sv.Detections.from_yolov8(results)
    detections = detections[(detections.class_id == 0)
                            & (detections.confidence > 0.4)]
    zone.trigger(detections, personPerFrame)

    # annotate  
    box_annotator = sv.BoxAnnotator(
        thickness=4, text_thickness=4, text_scale=2)
    labels = [f"{model.names[class_id]} {confidence:0.2f}" for _,
              confidence, class_id, _ in detections]
    frame = box_annotator.annotate(
        scene=frame, detections=detections, labels=labels)
    frame = zone_annotator.annotate(res, scene=frame,)

    return frame


# initiate polygon zone
polygon = np.array([
    [961, 1078],
    [763, 550],
    [1502, 254],
    [1914, 260],
    [1920, 1080]
])
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

zone = sv.PolygonZone(
    polygon=polygon, frame_resolution_wh=video_info.resolution_wh)

# initiate annotators
box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
zone_annotator = sv.PolygonZoneAnnotator(
    zone=zone, color=sv.Color.white(), thickness=6, text_thickness=3, text_scale=2)


generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

iterator = iter(generator)
frame = next(iterator)

out = cv2.VideoWriter(TARGET_VIDEO_PATH, cv2.VideoWriter_fourcc(
    *'mp4v'), 60, (WIDTH//2, HEIGHT//2))
num = 0

for frame in iterator:
    num += 1
    current = process_frame(frame, 1,res)

    if i < N:
        sumPerson += personPerFrame[-1]
        i += 1
    else:
        numPerson = sumPerson/N
        if numPerson-numPerson//1 < 0.5:
            personPerNFrame.append(numPerson//1)
        else:
            personPerNFrame.append(numPerson//1+1)
        sumPerson = 0
        i = 0

        currentFrame = personPerNFrame[-1]
        previousFrame = personPerNFrame[-2]
        if currentFrame < previousFrame:
            res = max(res, previousFrame)
        else:
            res += currentFrame-previousFrame

    width = int(current.shape[1] * 0.5)
    height = int(current.shape[0] * 0.5)
    new_size = (width, height)

    resized_img = cv2.resize(current, new_size)
    out.write(resized_img)
    cv2.imshow('Pedestrian Detection', resized_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
out.release()
cv2.destroyAllWindows()
