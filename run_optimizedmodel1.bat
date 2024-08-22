@echo off
REM Check if we're already in the correct environment
if "%CONDA_DEFAULT_ENV%" neq "avsr2" (
    REM Set the path to your Anaconda installation
    set ANACONDA_PATH=C:\Users\Bondo\miniconda3

    REM Activate the Anaconda base environment
    call "%ANACONDA_PATH%\Scripts\activate.bat"

    REM Activate your specific Conda environment
    call conda activate avsr2
)

REM Set CUDA optimizations
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
set CUDA_STREAM_PRIORITY=high

REM Preload CUDA
set CUDA_CACHE_PATH=%TEMP%\cuda_cache

REM Change to the directory containing your Python script
cd /d "%~dp0"

REM Run the optimized Python script with arguments
python demo_optimized.py data.modality=video pretrained_model_path=vsr_trlrwlrs2lrs3vox2avsp_base.pth file_path=Golf720v-1.mp4

