import sys
import os
import json
import base64
import numpy as np
import cv2
import torch
import uvicorn
import mediapipe as mp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from collections import deque

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from nn.model import STGCN
except ImportError:
    sys.exit(1)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "stgcn_posture_model.pth")
WINDOW_SIZE = 50

print(f"Loading model on {DEVICE}...")
model = STGCN(num_classes=3, in_channels=6)
if os.path.exists(MODEL_PATH):
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        print("✅ Model loaded.")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
else:
    print(f"⚠️ WARNING: Model not found at {MODEL_PATH}")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
MP_TO_17_MAP = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# --- PARANOID IMAGE PROCESSING ---
def extract_features(image_bgr):
    try:
        # 1. Convert to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # 2. FORCE MEMORY CONTIGUITY (The Fix for NORM_RECT error)
        # MediaPipe C++ sometimes fails if numpy array is non-contiguous
        if not image_rgb.flags['C_CONTIGUOUS']:
            image_rgb = np.ascontiguousarray(image_rgb)
            
        # 3. Explicit Process
        results = pose.process(image_rgb)
        
        if not results.pose_world_landmarks:
            return None
        
        coords = []
        for idx in MP_TO_17_MAP:
            lm = results.pose_world_landmarks.landmark[idx]
            coords.append([lm.x, lm.y, lm.z])
        return np.array(coords, dtype=np.float32)
    except Exception as e:
        # Log error but DO NOT crash
        print(f"⚠️ MediaPipe Error: {e}")
        return None

def normalize_skeleton(landmarks):
    mid_hip = (landmarks[11] + landmarks[12]) / 2.0
    return landmarks - mid_hip

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected.")
    
    frame_buffer = deque(maxlen=WINDOW_SIZE)
    previous_skeleton = None
    
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            
            if "frame" not in payload: continue

            # 1. Decode
            try:
                encoded_data = payload["frame"].split(',')[1]
                nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except:
                print("Decode failed")
                continue

            # 2. Strict Dimensions Check
            if img is None: 
                continue
            
            h, w, _ = img.shape
            
            # Reject small/empty frames immediately
            if w < 64 or h < 64: 
                print(f"Skipping tiny frame: {w}x{h}")
                continue

            # 3. Extract Pose (Now failsafe)
            current_skel = extract_features(img)
            
            response = {"status": "initializing", "confidence": 0.0, "risk_factors": []}

            if current_skel is not None:
                norm_skel = normalize_skeleton(current_skel)
                if previous_skeleton is None:
                    velocity = np.zeros_like(norm_skel)
                else:
                    velocity = norm_skel - previous_skeleton
                previous_skeleton = norm_skel
                
                feature = np.concatenate((norm_skel, velocity), axis=1)
                frame_buffer.append(feature)
                
                if len(frame_buffer) == WINDOW_SIZE:
                    input_np = np.array(frame_buffer)
                    input_tensor = torch.tensor(input_np, dtype=torch.float32).to(DEVICE)
                    input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)
                    
                    with torch.no_grad():
                        logits = model(input_tensor)
                        probs = torch.softmax(logits, dim=1)
                        pred_idx = torch.argmax(probs, dim=1).item()
                        confidence = probs[0, pred_idx].item()
                    
                    classes = ["safe", "warning", "critical"]
                    status = classes[pred_idx]
                    
                    risks = []
                    if status == "warning": risks = ["Bad Posture"]
                    if status == "critical": risks = ["Ergonomic Risk!"]
                    
                    response = {
                        "status": status,
                        "confidence": round(confidence, 2),
                        "risk_factors": risks
                    }

            await websocket.send_json(response)

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"Fatal Socket Error: {e}")