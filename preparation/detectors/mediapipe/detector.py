import mediapipe as mp
import numpy as np

class LandmarksDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

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
            
            # Extract specific landmarks that correspond to RetinaFace's 68 landmarks
            landmark_indices = [
                162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389, 71, 63,
                105, 66, 107, 336, 296, 334, 293, 301, 168, 197, 5, 4, 75, 97, 2, 326, 305, 33, 160, 158, 133,
                153, 144, 362, 385, 387, 263, 373, 380, 61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
                78, 82, 13, 312, 308, 317, 14, 87
            ]
            landmark_points = np.array([[int(face_landmarks[idx].x * iw), int(face_landmarks[idx].y * ih)] for idx in landmark_indices])
            
            landmarks.append(landmark_points)
        
        return landmarks