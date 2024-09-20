import os
import hydra
import torch
import torchaudio
import torchvision
from datamodule.transforms import AudioTransform, VideoTransform
from datamodule.av_dataset import cut_or_pad
from torch.cuda.amp import autocast, GradScaler
import cProfile
import pstats
import io
import time
import hashlib
from collections import OrderedDict
import json
import traceback

# Add the directory containing the script to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Change the working directory to the script's directory
os.chdir(script_dir)

class LimitedSizeDict(OrderedDict):
    def __init__(self, *args, **kwds):
        self.size_limit = kwds.pop("size_limit", None)
        OrderedDict.__init__(self, *args, **kwds)
        self._check_size_limit()

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        self._check_size_limit()

    def _check_size_limit(self):
        if self.size_limit is not None:
            while len(self) > self.size_limit:
                self.popitem(last=False)

class InferencePipeline(torch.nn.Module):
    def __init__(self, cfg, detector="retinaface"):
        super(InferencePipeline, self).__init__()
        self.modality = cfg.data.modality
        self.device = torch.device("cpu")
        self.device_string = "cpu"
        
        if self.modality in ["audio", "audiovisual"]:
            self.audio_transform = AudioTransform(subset="test")
        if self.modality in ["video", "audiovisual"]:
            if detector == "mediapipe":
                from preparation.detectors.mediapipe.detector import LandmarksDetector
                from preparation.detectors.mediapipe.video_process import VideoProcess
                self.landmarks_detector = LandmarksDetector()
                self.video_process = VideoProcess(convert_gray=False)
            elif detector == "retinaface":
                from preparation.detectors.retinaface.detector import LandmarksDetector
                from preparation.detectors.retinaface.video_process import VideoProcess
                self.landmarks_detector = LandmarksDetector(device=self.device_string)
                self.video_process = VideoProcess(convert_gray=False)
            self.video_transform = VideoTransform(subset="test")

        if cfg.data.modality in ["audio", "video"]:
            from lightning import ModelModule
        elif cfg.data.modality == "audiovisual":
            from lightning_av import ModelModule
        
        self.modelmodule = ModelModule(cfg).to(self.device)
        self.modelmodule.model.load_state_dict(torch.load(cfg.pretrained_model_path, map_location=self.device))
        self.modelmodule.eval()

        # Initialize mixed precision scaler
        self.scaler = GradScaler()

        # Initialize landmark cache with a limit of 10 entries
        self.landmark_cache = LimitedSizeDict(size_limit=10)

    @torch.no_grad()
    def forward(self, data_filename):
        data_filename = os.path.abspath(data_filename)
        assert os.path.isfile(data_filename), f"data_filename: {data_filename} does not exist."

        if self.modality in ["audio", "audiovisual"]:
            audio = self.load_and_process_audio(data_filename)
        
        if self.modality in ["video", "audiovisual"]:
            video = self.load_and_process_video(data_filename)

        with autocast():
            if self.modality == "video":
                transcript = self.modelmodule(video)
            elif self.modality == "audio":
                transcript = self.modelmodule(audio)
            elif self.modality == "audiovisual":
                print(len(audio), len(video))
                assert 530 < len(audio) // len(video) < 670, "The video frame rate should be between 24 and 30 fps."

                rate_ratio = len(audio) // len(video)
                if rate_ratio != 640:
                    print(f"The ideal video frame rate is set to 25 fps, but the current frame rate ratio, calculated as {len(video)*16000/len(audio):.1f}, which may affect the performance.")
                    audio = cut_or_pad(audio, len(video) * 640)
                transcript = self.modelmodule(video, audio)

        return transcript

    def load_and_process_audio(self, data_filename):
        waveform, sample_rate = torchaudio.load(data_filename, normalize=True)
        waveform = self.audio_process(waveform, sample_rate)
        waveform = waveform.transpose(1, 0)
        waveform = self.audio_transform(waveform)
        return waveform.to(self.device)

    def load_and_process_video(self, data_filename):
        video = torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()
        
        # Generate a unique key for this video
        video_hash = hashlib.md5(video.tobytes()).hexdigest()
        
        # Check if landmarks are cached
        if video_hash in self.landmark_cache:
            landmarks = self.landmark_cache[video_hash]
            print("Using cached landmarks")
        else:
            landmarks = self.landmarks_detector(video)
            self.landmark_cache[video_hash] = landmarks
            print("Computed and cached new landmarks")
        
        video = self.video_process(video, landmarks)
        video = torch.tensor(video)
        video = video.permute((0, 3, 1, 2))
        video = self.video_transform(video)
        return video.to(self.device)

    def audio_process(self, waveform, sample_rate, target_sample_rate=16000):
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, target_sample_rate
            )
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg):
    pr = cProfile.Profile()
    pr.enable()
    
    start_time = time.time()
    
    try:
        pipeline = InferencePipeline(cfg)
        transcript = pipeline(cfg.file_path)
        end_time = time.time()
        
        # Write transcript to a JSON file in C:\Users\Bondo\avsr2
        output_dir = r'C:\Users\Bondo\avsr2'
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'transcript.json')
        
        with open(output_file, 'w') as f:
            json.dump({
                'status': 'success',
                'transcript': transcript,
                'execution_time': end_time - start_time
            }, f)
        
        print(f"Transcript written to {output_file}")
        print(f"transcript: {transcript}")
        print(f"Total execution time: {end_time - start_time:.2f} seconds")

    except Exception as e:
        end_time = time.time()
        error_message = str(e)
        error_traceback = traceback.format_exc()
        
        output_dir = r'C:\Users\Bondo\avsr2'
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'transcript.json')
        
        with open(output_file, 'w') as f:
            json.dump({
                'status': 'error',
                'error_message': error_message,
                'error_traceback': error_traceback,
                'execution_time': end_time - start_time
            }, f)
        
        print(f"Error occurred. Details written to {output_file}")

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(10)  # Print only the top 10 time-consuming functions
    
    print("\nTop 10 time-consuming functions:")
    print(s.getvalue())

if __name__ == "__main__":
    main()

def load_and_process_video(self, data_filename):
    video = torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()
    
    # Generate a unique key for this video
    video_hash = hashlib.md5(video.tobytes()).hexdigest()
    
    # Check if landmarks are cached
    if video_hash in self.landmark_cache:
        landmarks = self.landmark_cache[video_hash]
        print("Using cached landmarks")
    else:
        landmarks = self.landmarks_detector(video)
        self.landmark_cache[video_hash] = landmarks
        print("Computed and cached new landmarks")
    
    video = self.video_process(video, landmarks)
    video = torch.tensor(video)
    video = video.permute((0, 3, 1, 2))
    video = self.video_transform(video)
    return video.to(self.device)