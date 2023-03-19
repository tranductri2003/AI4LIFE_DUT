from IPython import display
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from ultralytics import YOLO
import tkinter
import supervision as sv
import cv2


def show_Coordinate(frame):
    x = np.linspace(0, 20, 100)
    mpl.use('TkAgg')
    plt.show()
    sv.show_frame_in_notebook(frame, (16, 16))



