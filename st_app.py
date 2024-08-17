import cv2
from model import model
from cvzone.FaceMeshModule import FaceMeshDetector
import streamlit as st
import pandas as pd

st.title("Pendeteksi Mata Lelah")

# Initialize session state for cam_switch and status
if 'cam_switch' not in st.session_state:
    st.session_state['cam_switch'] = False

if 'status' not in st.session_state:
    st.session_state['status'] = "-"

if 'left_eye' not in st.session_state:
    st.session_state['left_eye'] = 0

if 'right_eye' not in st.session_state:
    st.session_state['right_eye'] = 0

# Define the UI elements
cam_switch = st.checkbox("Video Capture", key='cam_switch')

status = st.text(f"Status: {st.session_state['status']}")
left_eye = st.text(f"Mata Kanan: {st.session_state['left_eye']}")
right_eye = st.text(f"Mata Kanan: {st.session_state['right_eye']}")

def update():
    status.text(f"Status: {st.session_state['status']}")
    left_eye.text(f"Mata Kanan: {st.session_state['left_eye']}")
    right_eye.text(f"Mata Kanan: {st.session_state['right_eye']}")

img_capture = st.empty()

detector = FaceMeshDetector(maxFaces=1)
ids = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]

if cam_switch:
    cap = cv2.VideoCapture(0)
    eye_heights = []
    
    while st.session_state['cam_switch']:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if len(eye_heights) >= 5:
            data = {}
        
            for i, eye_height in enumerate(eye_heights):
                for i in range(5):
                    data[f"mata_kiri{i+1}"] = [eye_height[0]]
                for i in range(5):
                    data[f"mata_kanan{i+1}"] = [eye_height[1]]
            
            f = pd.DataFrame(data)
            pred = model.predict(f)
            st.session_state['status'] = "Lelah" if pred[0] == 1 else "Tidak Lelah"
            update()
            eye_heights = []
        
        img, faces = detector.findFaceMesh(frame, draw=False)
    
        if faces:
            face = faces[0]
            for id in ids:
                cv2.circle(img, face[id], 5, (0, 0, 255), cv2.FILLED)

                leftUp = face[159]
                leftDown = face[23]
                leftLeft = face[130]
                leftRight = face[243]
                leftLengthVer, _ = detector.findDistance(leftUp, leftDown)
            
                rightUp = face[386]
                rightDown = face[374]
                rightLeft = face[382]
                rightRight = face[263]
                rightLengthVer, _ = detector.findDistance(rightUp, rightDown)

                cv2.line(img, leftUp, leftDown, (0, 200, 0), 3)
                cv2.line(img, rightUp, rightDown, (0, 200, 0), 3)
                
                st.session_state['left_eye'] = leftLengthVer
                st.session_state['right_eye'] = rightLengthVer
                
                update()
                 
                eye_heights.append((leftLengthVer, rightLengthVer))
        
        img_capture.image(img, use_column_width=True)
    
    cap.release()
else:
    img_capture.empty()
