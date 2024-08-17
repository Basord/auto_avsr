#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import sys
import os

print(f"Current working directory: {os.getcwd()}")
print(f"Contents of current directory: {os.listdir('.')}")
print(f"Contents of preparation directory: {os.listdir('preparation')}")

import warnings
import sys
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
except ImportError as e:
    print(f"Error importing torch: {e}")

try:
    from face_alignment import FaceAlignment, LandmarksType
    print("Successfully imported face_alignment")
except ImportError as e:
    print(f"Error importing face_alignment: {e}")

warnings.filterwarnings("ignore")

class LandmarksDetector:
    def __init__(self, device="cpu", model_name="resnet50"):
        self.device = device
        self.face_alignment = FaceAlignment(LandmarksType.TWO_D, device=device)
        
    def __call__(self, video_frames):
        landmarks = []
        for frame in video_frames:
            # FaceAlignment can detect faces and predict landmarks in one step
            face_landmarks = self.face_alignment.get_landmarks(frame)
            
            if face_landmarks is None or len(face_landmarks) == 0:
                landmarks.append(None)
            else:
                # If multiple faces are detected, choose the largest one
                if len(face_landmarks) > 1:
                    max_id = max(range(len(face_landmarks)), 
                                 key=lambda i: (face_landmarks[i][:,0].max() - face_landmarks[i][:,0].min()) * 
                                               (face_landmarks[i][:,1].max() - face_landmarks[i][:,1].min()))
                    landmarks.append(face_landmarks[max_id])
                else:
                    landmarks.append(face_landmarks[0])
        
        return landmarks
