import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import numpy as np
from pycaw.pycaw import AudioUtilities
import urllib.request
import os

# --- Download hand landmarker model if not present ---
model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        model_path
    )

# --- PyCaw Setup ---
device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

# --- MediaPipe New Tasks API Setup ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7
)

cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        success, img = cap.read()
        if not success:
            break

        # Convert to MediaPipe Image
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)

        # Detect landmarks
        result = landmarker.detect(mp_image)

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                h, w, _ = img.shape

                # Draw landmarks manually
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 4, (0, 255, 0), cv2.FILLED)

                # Thumb tip = index 4, Index finger tip = index 8
                x1 = int(hand_landmarks[4].x * w)
                y1 = int(hand_landmarks[4].y * h)
                x2 = int(hand_landmarks[8].x * w)
                y2 = int(hand_landmarks[8].y * h)

                length = math.hypot(x2 - x1, y2 - y1)

                # Map distance to volume
                masterVol = np.interp(length, [20, 200], [minVol, maxVol])
                volume.SetMasterVolumeLevel(masterVol, None)

                # Visuals
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                if length < 30:
                    cv2.circle(img, (int((x1+x2)/2), int((y1+y2)/2)), 10, (255, 0, 0), cv2.FILLED)

        cv2.imshow("Hand Gesture Volume Control", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()