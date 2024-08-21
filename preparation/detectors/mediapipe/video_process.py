import os
import cv2
import numpy as np
from skimage import transform as tf

def linear_interpolate(landmarks, start_idx, stop_idx):
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx - start_idx):
        landmarks[start_idx + idx] = (
            start_landmarks + idx / float(stop_idx - start_idx) * delta
        )
    return landmarks

def cut_patch(img, landmarks, height, width, threshold=5):
    center_x, center_y = np.mean(landmarks, axis=0)
    if abs(center_y - img.shape[0] / 2) > height + threshold:
        raise OverflowError("too much bias in height")
    if abs(center_x - img.shape[1] / 2) > width + threshold:
        raise OverflowError("too much bias in width")
    y_min = int(round(np.clip(center_y - height, 0, img.shape[0])))
    y_max = int(round(np.clip(center_y + height, 0, img.shape[0])))
    x_min = int(round(np.clip(center_x - width, 0, img.shape[1])))
    x_max = int(round(np.clip(center_x + width, 0, img.shape[1])))
    cutted_img = np.copy(img[y_min:y_max, x_min:x_max])
    return cutted_img

class VideoProcess:
    def __init__(
        self,
        mean_face_path="20words_mean_face.npy",
        crop_width=96,
        crop_height=96,
        start_idx=48,
        stop_idx=68,
        window_margin=12,
        convert_gray=True,
    ):
        self.reference = np.load(
            os.path.join(os.path.dirname(__file__), mean_face_path)
        )
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.start_idx = start_idx
        self.stop_idx = stop_idx
        self.window_margin = window_margin
        self.convert_gray = convert_gray

    def __call__(self, video, landmarks):
        preprocessed_landmarks = self.interpolate_landmarks(landmarks)
        if (
            not preprocessed_landmarks
            or len(preprocessed_landmarks) < self.window_margin
        ):
            return
        sequence = self.crop_patch(video, preprocessed_landmarks)
        assert sequence is not None, "crop an empty patch."
        return sequence

    def crop_patch(self, video, landmarks):
        sequence = []
        for frame_idx, frame in enumerate(video):
            window_margin = min(
                self.window_margin // 2, frame_idx, len(landmarks) - 1 - frame_idx
            )
            smoothed_landmarks = np.mean(
                [
                    landmarks[x]
                    for x in range(
                        frame_idx - window_margin, frame_idx + window_margin + 1
                    )
                ],
                axis=0,
            )
            smoothed_landmarks += landmarks[frame_idx].mean(
                axis=0
            ) - smoothed_landmarks.mean(axis=0)
            transformed_frame, transformed_landmarks = self.affine_transform(
                frame, smoothed_landmarks, self.reference, grayscale=self.convert_gray
            )
            
            # Adjust the landmark range to focus on the mouth region
            mouth_landmarks = transformed_landmarks[self.start_idx:self.stop_idx]
            
            # Calculate the bounding box for the mouth region
            x_min, y_min = np.min(mouth_landmarks, axis=0)
            x_max, y_max = np.max(mouth_landmarks, axis=0)
            
            # Add some margin around the mouth region
            margin = 10
            y_min = max(0, y_min - margin)
            y_max = min(transformed_frame.shape[0], y_max + margin)
            x_min = max(0, x_min - margin)
            x_max = min(transformed_frame.shape[1], x_max + margin)
            
            # Crop the patch
            patch = transformed_frame[int(y_min):int(y_max), int(x_min):int(x_max)]
            
            # Resize the patch to the desired dimensions
            patch = cv2.resize(patch, (self.crop_width, self.crop_height))
            
            sequence.append(patch)
        return np.array(sequence)

    def interpolate_landmarks(self, landmarks):
        valid_frames_idx = [idx for idx, lm in enumerate(landmarks) if lm is not None]

        if not valid_frames_idx:
            return None

        for idx in range(1, len(valid_frames_idx)):
            if valid_frames_idx[idx] - valid_frames_idx[idx - 1] > 1:
                landmarks = linear_interpolate(
                    landmarks, valid_frames_idx[idx - 1], valid_frames_idx[idx]
                )

        valid_frames_idx = [idx for idx, lm in enumerate(landmarks) if lm is not None]

        if valid_frames_idx:
            landmarks[: valid_frames_idx[0]] = [
                landmarks[valid_frames_idx[0]]
            ] * valid_frames_idx[0]
            landmarks[valid_frames_idx[-1] :] = [landmarks[valid_frames_idx[-1]]] * (
                len(landmarks) - valid_frames_idx[-1]
            )

        assert all(lm is not None for lm in landmarks), "not every frame has landmark"

        return landmarks

    def affine_transform(
        self,
        frame,
        landmarks,
        reference,
        grayscale=True,
        target_size=(256, 256),
        reference_size=(256, 256),
        stable_points=(28, 33, 36, 39, 42, 45, 48, 54),
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        border_value=0,
    ):
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        stable_reference = self.get_stable_reference(
            reference, stable_points, reference_size, target_size
        )
        transform = self.estimate_affine_transform(
            landmarks, stable_points, stable_reference
        )
        transformed_frame, transformed_landmarks = self.apply_affine_transform(
            frame,
            landmarks,
            transform,
            target_size,
            interpolation,
            border_mode,
            border_value,
        )

        return transformed_frame, transformed_landmarks

    def get_stable_reference(
        self, reference, stable_points, reference_size, target_size
    ):
        stable_reference = np.vstack([reference[x] for x in stable_points])
        stable_reference[:, 0] -= (reference_size[0] - target_size[0]) / 2.0
        stable_reference[:, 1] -= (reference_size[1] - target_size[1]) / 2.0
        return stable_reference

    def estimate_affine_transform(self, landmarks, stable_points, stable_reference):
        return cv2.estimateAffinePartial2D(
            np.vstack([landmarks[x] for x in stable_points]),
            stable_reference,
            method=cv2.LMEDS,
        )[0]

    def apply_affine_transform(
        self,
        frame,
        landmarks,
        transform,
        target_size,
        interpolation,
        border_mode,
        border_value,
    ):
        transformed_frame = cv2.warpAffine(
            frame,
            transform,
            dsize=(target_size[0], target_size[1]),
            flags=interpolation,
            borderMode=border_mode,
            borderValue=border_value,
        )
        transformed_landmarks = (
            np.matmul(landmarks, transform[:, :2].transpose())
            + transform[:, 2].transpose()
        )
        return transformed_frame, transformed_landmarks