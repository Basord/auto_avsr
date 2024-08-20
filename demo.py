import warnings
import torch
import numpy as np
from face_alignment import FaceAlignment, LandmarksType

warnings.filterwarnings("ignore")

class LandmarksDetector:
    def __init__(self, device="cuda:0", batch_size=32):
        self.device = device
        self.batch_size = batch_size
        self.face_alignment = FaceAlignment(LandmarksType.TWO_D, device=device, flip_input=False)
        
        # Use DataParallel if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            self.face_alignment = torch.nn.DataParallel(self.face_alignment)
        
    def __call__(self, video_frames):
        landmarks = []
        
        # Convert video_frames to torch tensor and move to GPU
        video_frames = torch.from_numpy(video_frames).to(self.device)
        
        # Process frames in batches
        for i in range(0, len(video_frames), self.batch_size):
            batch = video_frames[i:i+self.batch_size]
            batch_landmarks = self.face_alignment.get_landmarks_from_batch(batch)
            
            for face_landmarks in batch_landmarks:
                if face_landmarks is None or len(face_landmarks) == 0:
                    landmarks.append(None)
                else:
                    if len(face_landmarks) > 1:
                        max_id = max(range(len(face_landmarks)), 
                                     key=lambda i: (face_landmarks[i][:,0].max() - face_landmarks[i][:,0].min()) * 
                                                   (face_landmarks[i][:,1].max() - face_landmarks[i][:,1].min()))
                        landmarks.append(face_landmarks[max_id])
                    else:
                        landmarks.append(face_landmarks[0])
        
        return landmarks

def apply_landmarks_to_frame(frame, landmarks):
    # Implement your logic to apply landmarks to the frame
    return frame

def load_and_process_video(video_frames, device="cuda:0", batch_size=128):
    detector = LandmarksDetector(device=device, batch_size=batch_size)
    landmarks = detector(video_frames)
    processed_frames = [apply_landmarks_to_frame(frame, lm) for frame, lm in zip(video_frames, landmarks)]
    return processed_frames