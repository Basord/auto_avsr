import warnings
import torch
from face_alignment import FaceAlignment, LandmarksType
import numpy as np
from multiprocessing import Pool, cpu_count, shared_memory
from functools import partial

warnings.filterwarnings("ignore")

def process_batch(shared_mem_name, shape, dtype, batch_indices, device):
    # Access shared memory
    existing_shm = shared_memory.SharedMemory(name=shared_mem_name)
    video_frames = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    
    # Create FaceAlignment model inside the process
    face_alignment = FaceAlignment(LandmarksType.TWO_D, device=device)
    
    landmarks = []
    for i in batch_indices:
        frame = video_frames[i]
        face_landmarks = face_alignment.get_landmarks(frame)
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
    
    existing_shm.close()
    return landmarks

class LandmarksDetector:
    def __init__(self, device="cuda:0", batch_size=32, num_workers=None):
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else cpu_count()
        
    def __call__(self, video_frames):
        if len(video_frames) > 1000:
            video_frames = video_frames[::len(video_frames)//1000]

        # Create shared memory
        shm = shared_memory.SharedMemory(create=True, size=video_frames.nbytes)
        shared_frames = np.ndarray(video_frames.shape, dtype=video_frames.dtype, buffer=shm.buf)
        shared_frames[:] = video_frames[:]  # Copy the data to shared memory

        batch_indices = [range(i, min(i + self.batch_size, len(video_frames))) 
                         for i in range(0, len(video_frames), self.batch_size)]

        with Pool(processes=self.num_workers) as pool:
            landmarks = pool.map(partial(process_batch, 
                                         shm.name, 
                                         video_frames.shape, 
                                         video_frames.dtype, 
                                         device=self.device), 
                                 batch_indices)

        # Clean up
        shm.close()
        shm.unlink()

        return [item for sublist in landmarks for item in sublist]

def apply_landmarks_to_frame(frame, landmarks):
    # Implement your logic to apply landmarks to the frame
    return frame

def process_video_chunk(shared_mem_name, shape, dtype, chunk_indices, device):
    existing_shm = shared_memory.SharedMemory(name=shared_mem_name)
    video_frames = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    
    chunk = video_frames[chunk_indices]
    detector = LandmarksDetector(device=device)
    landmarks = detector(chunk)
    processed_frames = [apply_landmarks_to_frame(frame, lm) for frame, lm in zip(chunk, landmarks)]
    
    existing_shm.close()
    return processed_frames

def load_and_process_video(video_frames, device="cuda:0", num_workers=None):
    if num_workers is None:
        num_workers = cpu_count()

    chunk_size = 100  # Adjust as needed
    
    # Create shared memory
    shm = shared_memory.SharedMemory(create=True, size=video_frames.nbytes)
    shared_frames = np.ndarray(video_frames.shape, dtype=video_frames.dtype, buffer=shm.buf)
    shared_frames[:] = video_frames[:]  # Copy the data to shared memory

    chunk_indices = [range(i, min(i + chunk_size, len(video_frames))) 
                     for i in range(0, len(video_frames), chunk_size)]
    
    with Pool(processes=num_workers) as pool:
        processed_chunks = pool.map(partial(process_video_chunk, 
                                            shm.name, 
                                            video_frames.shape, 
                                            video_frames.dtype, 
                                            device=device), 
                                    chunk_indices)
    
    # Clean up
    shm.close()
    shm.unlink()
    
    return [frame for chunk in processed_chunks for frame in chunk]