'''
Command to install libraries:
pip3 install opencv-python mediapipe numpy

'''
import cv2
import mediapipe as mp
import numpy as np
import subprocess
from math import hypot

mp_hands = mp.solutions.hands 
my_hands = mp_hands.Hands() 
mp_draw = mp.solutions.drawing_utils 

cap = cv2.VideoCapture(1)

def set_volume(volume_level):
    volume_level = int(np.clip(volume_level, 0, 100))  
    subprocess.run(["osascript", "-e", f"set volume output volume {volume_level}"]) 

while True: 
    ret, frame = cap.read() 
    if not ret: 
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    results = my_hands.process(frame_rgb) 

    lm_list = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if lm_list:
        x1, y1 = lm_list[4][1], lm_list[4][2]
        x2, y2 = lm_list[8][1], lm_list[8][2]

        cv2.circle(frame, (x1, y1), 10, (255, 10, 0), cv2.FILLED)
        cv2.circle(frame, (x2, y2), 10, (255, 0, 0), cv2.FILLED)

        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 10), 3)

        # Distance between the tips
        length = hypot(x2 - x1, y2 - y1)

        vol = np.interp(length, [15, 220], [0, 100])  
        set_volume(vol)

        vol_percentage = np.interp(length, [15, 220], [0, 100])
        cv2.putText(frame, f'{int(vol_percentage)}%', (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (244, 208, 63), 2)

    cv2.imshow('Volume Control System', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
