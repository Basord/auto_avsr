import mediapipe as mp
import numpy as np

class LandmarksDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

    def __call__(self, video_frames):
        return self.detect(video_frames)

    def detect(self, video_frames):
        landmarks = []
        for frame in video_frames:
            results = self.face_mesh.process(frame)
            if not results.multi_face_landmarks:
                landmarks.append(None)
                continue
            
            face_landmarks = results.multi_face_landmarks[0].landmark
            ih, iw, _ = frame.shape
            
            # Extract key points similar to RetinaFace (eyes, nose, mouth)
            key_points = [
                [int(face_landmarks[33].x * iw), int(face_landmarks[33].y * ih)],  # Left eye
                [int(face_landmarks[263].x * iw), int(face_landmarks[263].y * ih)],  # Right eye
                [int(face_landmarks[1].x * iw), int(face_landmarks[1].y * ih)],  # Nose
                [int(face_landmarks[61].x * iw), int(face_landmarks[61].y * ih)],  # Mouth left
                [int(face_landmarks[291].x * iw), int(face_landmarks[291].y * ih)]  # Mouth right
            ]
            
            landmarks.append(np.array(key_points))
        
        return landmarks