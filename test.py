import streamlit as st
import cv2
import mediapipe as mp
import time
import numpy as np
import math

NUM_FACE = 1
thresh = 0.2  # You need to define the threshold value for eye aspect ratio
flag = 0  # Initialize the flag for drowsiness detection

st.title("Digital Eye Strain Detector")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(1)  # Use 1 for default camera

class FaceLandMarks():
    def __init__(self, staticMode=False, maxFace=NUM_FACE, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFace = maxFace
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon
        