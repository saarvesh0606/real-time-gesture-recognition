from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp
import time
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER
import random
import math

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

camera = cv2.VideoCapture(0)

THUMB_TIP = 4
THUMB_IP  = 3
INDEX_TIP = 8
INDEX_PIP = 6
MIDDLE_TIP = 12
MIDDLE_PIP = 10
RING_TIP = 16
RING_PIP = 14
PINKY_TIP = 20
PINKY_PIP = 18



def get_finger_states(hand_landmarks, handedness_label, margin=0.02):
    lm = hand_landmarks.landmark

    
    index_up  = lm[INDEX_TIP].y  < lm[INDEX_PIP].y  - margin
    middle_up = lm[MIDDLE_TIP].y < lm[MIDDLE_PIP].y - margin
    ring_up   = lm[RING_TIP].y   < lm[RING_PIP].y   - margin
    pinky_up  = lm[PINKY_TIP].y  < lm[PINKY_PIP].y  - margin

    
    if handedness_label == "Right":
        thumb_up = lm[THUMB_TIP].x < lm[THUMB_IP].x - margin
    else:  # "Left"
        thumb_up = lm[THUMB_TIP].x > lm[THUMB_IP].x + margin

    return {
        "thumb": thumb_up,
        "index": index_up,
        "middle": middle_up,
        "ring": ring_up,
        "pinky": pinky_up,
    }

def classify_rps(fingers):
    thumb  = fingers["thumb"]
    index  = fingers["index"]
    middle = fingers["middle"]
    ring   = fingers["ring"]
    pinky  = fingers["pinky"]

   
    if not index and not middle and not ring and not pinky:
        return "Rock"

    
    if thumb and index and middle and ring and pinky:
        return "Paper"

    
    if index and middle and not ring and not pinky:
        return "Scissors"

    return "Unknown"



def generate_frames():
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=2
    ) as hands:
        
        prev_time = 0

        # State for AI (one-hand) mode
        countdown_end_time = None
        machine_move = None
        result_text = ""
        user_move = ""

        while True:
            success, frame = camera.read()
            if not success:
                break
            
            frame = cv2.flip(frame, 1)
            frame.flags.writeable = False
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = hands.process(image_rgb)
            
            frame.flags.writeable = True
            
            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
            # FPS calculation and display
            current_time = time.time()
            fps = 1 / (current_time - prev_time + 1e-5) 
            prev_time = current_time
            
            cv2.putText(
                frame, 
                f'FPS: {int(fps)}', 
                (20, 50), 
                cv2.FONT_HERSHEY_PLAIN, 
                2, 
                (0, 255, 0), 
                2
            )        
                
            # Handedness display (Left / Right)
            if results.multi_handedness and results.multi_hand_landmarks:
                for index, hand_handedness in enumerate(results.multi_handedness):
                    handedness_label = hand_handedness.classification[0].label
                    hand_landmarks = results.multi_hand_landmarks[index]
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    h, w, c = frame.shape
                    wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
                    
                    cv2.putText(
                        frame,
                        handedness_label,
                        (wrist_x - 30, wrist_y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        2
                    )
            
            # Hand open/closed status 
            if results.multi_handedness and results.multi_hand_landmarks:
                for index, hand_handedness in enumerate(results.multi_handedness):
                    hand_landmarks = results.multi_hand_landmarks[index]
                    finger_status = []
                    for tip_id in [4, 8, 12, 16, 20]:
                        finger_tip = hand_landmarks.landmark[tip_id]
                        finger_dip = hand_landmarks.landmark[tip_id - 2]
                        
                        # Thumb block (x-axis)
                        if tip_id == 4:
                            if finger_tip.x < finger_dip.x:
                                finger_status.append('Open')
                            else:
                                finger_status.append('Closed')
                        # Other fingers (y-axis)       
                        else:
                            if finger_tip.y < finger_dip.y:
                                finger_status.append('Open')
                            else:
                                finger_status.append('Closed')
                    
                    open_count = finger_status.count('Open')
                    hand_status = 'Open' if open_count >= 3 else 'Closed'
                    
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    h, w, c = frame.shape
                    wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
                    cv2.putText(
                        frame,
                        hand_status,
                        (wrist_x - 30, wrist_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )

            # game logic
            
            if results.multi_hand_landmarks and results.multi_handedness:
                hand_count = len(results.multi_hand_landmarks)

              
                if hand_count == 2:
                    # reset AI mode state
                    countdown_end_time = None
                    machine_move = None
                    result_text = ""
                    user_move = ""

                    hands_gestures = []

                    for hand_landmarks, handedness in zip(
                        results.multi_hand_landmarks,
                        results.multi_handedness
                    ):
                        handedness_label = handedness.classification[0].label  
                        fingers = get_finger_states(hand_landmarks, handedness_label)
                        current_gesture = classify_rps(fingers)
                        hands_gestures.append(current_gesture)

                    move1 = hands_gestures[0]
                    move2 = hands_gestures[1]
                    manual_result_text = "Unknown"

                    if move1 == "Unknown" or move2 == "Unknown":
                        manual_result_text = "Show Hands Clearly"
                    elif move1 == move2:
                        manual_result_text = "Tie"
                    elif (move1 == "Rock" and move2 == "Scissors") or \
                         (move1 == "Scissors" and move2 == "Paper") or \
                         (move1 == "Paper" and move2 == "Rock"):
                        manual_result_text = "Hand 1 Wins"
                    else:
                        manual_result_text = "Hand 2 Wins"

                    cv2.putText(
                        frame,  
                        f"{move1} vs {move2}: {manual_result_text}",
                        (50, 450),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 0),
                        1
                    )
                
                # with comp
                
                elif hand_count == 1:
                    now = time.time()

                    # countdoen
                    if countdown_end_time is None:
                        countdown_end_time = now + 3   # 3 seconds from now
                        machine_move = random.choice(['Rock', 'Paper', 'Scissors'])
                        result_text = ""
                        user_move = ""

                    # Countdown still running → show timer
                    if now < countdown_end_time:
                        remaining = int(countdown_end_time - now) + 1
                        cv2.putText(
                            frame,
                            f"Show your move in: {remaining}",
                            (100, 400),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2,
                            (0, 0, 255),
                            3
                        )
                    else:
                        # Countdown finished
                        if user_move == "":  
                            user_hand_landmarks = results.multi_hand_landmarks[0]
                            handedness_label = results.multi_handedness[0].classification[0].label
                            fingers = get_finger_states(user_hand_landmarks, handedness_label)
                            user_move = classify_rps(fingers)

                            if user_move == "Unknown":
                                result_text = "Show Hand Clearly"
                            elif user_move == machine_move:
                                result_text = "Tie"
                            elif (user_move == "Rock" and machine_move == "Scissors") or \
                                 (user_move == "Scissors" and machine_move == "Paper") or \
                                 (user_move == "Paper" and machine_move == "Rock"):
                                result_text = "You Win!"
                            else:
                                result_text = "Machine Wins!"

                        cv2.putText(
                            frame,
                            f"You: {user_move}  |  AI: {machine_move}  ->  {result_text}",
                            (50, 450),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.85,
                            (0, 255, 255),
                            1
                        )
                else:
                    # No hands visible -> reset AI state
                    countdown_end_time = None
                    machine_move = None
                    result_text = ""
                    user_move = ""

            else:
                # No hands at all -> reset AI state
                countdown_end_time = None
                machine_move = None
                result_text = ""
                user_move = ""
            
            # Encode frame and stream
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    try:
        app.run(debug=True)
    finally:
        camera.release()
        cv2.destroyAllWindows()
