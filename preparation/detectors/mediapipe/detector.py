import mediapipe as mp
import numpy as np
import threading
from queue import Queue

class LandmarksDetector:
    def __init__(self, batch_size=32, num_threads=4):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.batch_size = batch_size
        self.num_threads = num_threads

    def __call__(self, video_frames):
        return self.detect_parallel(video_frames)

    def detect_batch(self, batch, result_queue):
        with self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5
        ) as face_mesh:
            landmarks = []
            for frame in batch:
                results = face_mesh.process(frame)
                if not results.multi_face_landmarks:
                    landmarks.append(None)
                    continue
                
                face_landmarks = results.multi_face_landmarks[0].landmark
                ih, iw, _ = frame.shape
                
                key_points = [
                    [int(face_landmarks[33].x * iw), int(face_landmarks[33].y * ih)],
                    [int(face_landmarks[263].x * iw), int(face_landmarks[263].y * ih)],
                    [int(face_landmarks[1].x * iw), int(face_landmarks[1].y * ih)],
                    [int(face_landmarks[61].x * iw), int(face_landmarks[61].y * ih)],
                    [int(face_landmarks[291].x * iw), int(face_landmarks[291].y * ih)]
                ]
                
                landmarks.append(np.array(key_points))
            
            result_queue.put(landmarks)

    def detect_parallel(self, video_frames):
        all_landmarks = []
        threads = []
        result_queue = Queue()

        for i in range(0, len(video_frames), self.batch_size):
            batch = video_frames[i:i+self.batch_size]
            thread = threading.Thread(target=self.detect_batch, args=(batch, result_queue))
            threads.append(thread)
            thread.start()

            if len(threads) >= self.num_threads:
                for thread in threads:
                    thread.join()
                while not result_queue.empty():
                    all_landmarks.extend(result_queue.get())
                threads = []

        for thread in threads:
            thread.join()
        while not result_queue.empty():
            all_landmarks.extend(result_queue.get())

        return all_landmarks