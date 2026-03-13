# Audio Transcriber

Local-only desktop application that transcribes audio files using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) on your own machine. Designed for transcribing Spanish WhatsApp voice notes (`.ogg`), but supports other audio and video formats.

No cloud APIs. No internet required after the initial model download.

**Supported formats:** `.ogg`, `.mp3`, `.wav`, `.m4a`, `.mp4`

**Output formats:** `.txt` (plain text), `.srt` (subtitles), `.json` (structured metadata + segments)

---

## Requirements

- **Windows 10 or later**
- **Python 3.11+**
- **ffmpeg** on your system PATH
- **NVIDIA GPU + CUDA Toolkit 12.x + cuDNN 9.x** (optional, for GPU acceleration)

---

## 1. Install Python

Download Python 3.11 or newer from [python.org](https://www.python.org/downloads/).

During installation:

1. Check **"Add python.exe to PATH"** on the first screen.
2. Click **"Install Now"** (or customise if needed).

Verify after installation:

```
python --version
```

You should see `Python 3.11.x` or higher.

---

## 2. Install ffmpeg

ffmpeg is required for decoding audio files. faster-whisper calls it internally.

### Option A: winget (recommended)

```
winget install Gyan.FFmpeg
```

This installs ffmpeg and adds it to PATH automatically. You may need to restart your terminal.

### Option B: Manual install

1. Download a release build from [gyan.dev/ffmpeg](https://www.gyan.dev/ffmpeg/builds/) — get the **ffmpeg-release-essentials.zip**.
2. Extract the zip to a permanent location, e.g. `C:\ffmpeg`.
3. Add the `bin` folder to your system PATH:
   - Press `Win + R`, type `sysdm.cpl`, press Enter.
   - Go to **Advanced** → **Environment Variables**.
   - Under **System variables**, find `Path`, click **Edit**.
   - Click **New** and add `C:\ffmpeg\bin`.
   - Click **OK** on all dialogs.
4. Open a **new** terminal and verify:

```
ffmpeg -version
```

---

## 3. GPU Acceleration (Optional)

GPU acceleration is **optional**. The app works on CPU out of the box. If you have an NVIDIA GPU and want significantly faster transcription (especially with larger models), follow the steps below.

### 3.1 Install NVIDIA GPU Drivers

Make sure you have up-to-date NVIDIA drivers from [nvidia.com/drivers](https://www.nvidia.com/drivers).

### 3.2 Install CUDA Toolkit 12.x

1. Download and install from the [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive). Any CUDA 12.x release works (e.g. 12.6, 12.8, 12.9).
2. During installation, the default options are fine.
3. The installer typically sets the `CUDA_PATH` environment variable automatically (e.g. `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9`).

### 3.3 Install cuDNN 9.x

cuDNN is required by CTranslate2 for GPU inference.

1. Download cuDNN 9.x for CUDA 12 from [developer.nvidia.com/cudnn-downloads](https://developer.nvidia.com/cudnn-downloads) (requires a free NVIDIA developer account).
2. Install using the provided installer, or extract and copy the files into your CUDA Toolkit directory:
   - `bin\*.dll` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin\`
   - `include\*.h` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\include\`
   - `lib\x64\*.lib` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\lib\x64\`

### 3.4 Environment Variables

The app auto-detects the CUDA Toolkit location in this order:

1. `CUDA_PATH` environment variable (set by the CUDA installer)
2. `CUDA_HOME` environment variable
3. Default path: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`

If the CUDA installer set `CUDA_PATH` correctly, no extra configuration is needed.

> **Note on Python 3.8+ and DLL loading:** Python 3.8 changed how Windows loads DLLs — the system `PATH` is no longer searched by default. The app handles this automatically by calling `os.add_dll_directory()` for the CUDA `bin` directory. You do **not** need to manually add the CUDA `bin` directory to your PATH, but having `CUDA_PATH` set (or the toolkit installed in the default location) is required for auto-detection to work.

### 3.5 Verify CUDA Setup

```
python -c "import ctranslate2; print('CUDA devices:', ctranslate2.get_cuda_device_count())"
```

Should print `CUDA devices: 1` (or more). If it prints `0`, the CUDA toolkit or drivers are not installed correctly.

---

## 4. Set Up the Project

### Clone or download the project

Place the project folder somewhere convenient, e.g. `C:\Users\YourName\audio_transcriber`.

### Option A: Using uv (recommended)

[uv](https://docs.astral.sh/uv/) manages the virtual environment and dependencies automatically:

```
uv run python main.py
```

### Option B: Using pip

Create and activate a virtual environment:

```
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

This installs `faster-whisper` and its dependencies (including CTranslate2, cuBLAS, and cuDNN Python packages).

---

## 5. How Models Work

faster-whisper downloads Whisper models from Hugging Face the **first time** you use a given model size. After that, the model is cached locally and no internet is needed.

Default cache location on Windows:

```
C:\Users\YourName\.cache\huggingface\hub
```

Available model sizes (smallest → largest):

| Model | Disk Size | RAM (approx) | Speed | Accuracy |
|-------|-----------|--------------|-------|----------|
| tiny | ~75 MB | ~1 GB | Fastest | Lowest |
| base | ~150 MB | ~1 GB | Fast | Low |
| small | ~500 MB | ~2 GB | Moderate | Good |
| medium | ~1.5 GB | ~5 GB | Slow | High |
| large-v3 | ~3 GB | ~10 GB | Slowest | Highest |

For Spanish WhatsApp voice notes, **small** is a good default. Use **medium** or **large-v3** if you need better accuracy and have the RAM/VRAM.

---

## 6. Run the App

### With uv

```
uv run python main.py
```

### With pip (virtual environment activated)

```
python main.py
```

The GUI window will open. If ffmpeg is missing, you will see an error dialog immediately.

Settings are saved automatically on exit and restored on the next launch.

---

## 7. Example Workflow: Spanish WhatsApp Voice Notes

1. Export voice notes from WhatsApp to a folder on your PC (they are `.ogg` files).
2. Launch the app.
3. Click **Select Files** and pick one or more `.ogg` files.
4. Settings:
   - **Model:** small (or medium/large-v3 for better accuracy)
   - **Device:** auto
   - **Compute:** auto
   - **Language:** Spanish (forced)
   - **Output:** TXT checked (and/or SRT, JSON)
5. Click **Transcribe**.
6. Wait for the model to load (first run downloads it — needs internet).
7. Each file is transcribed sequentially. Progress shows in the UI.
8. Output files appear next to each source file (or in the chosen output folder):
   - `voice_note.txt` — plain transcript
   - `voice_note.srt` — timed subtitles
   - `voice_note.json` — metadata + segments

If an output file already exists, a numeric suffix is appended automatically (e.g. `voice_note_1.txt`).

---

## 8. CPU vs CUDA

| Setting | When to use |
|---------|-------------|
| **Device: auto** | Recommended. Uses CUDA if available, falls back to CPU. |
| **Device: cpu** | Force CPU. Works on all machines. Slower but reliable. |
| **Device: cuda** | Force CUDA. Requires an NVIDIA GPU with CUDA 12.x + cuDNN 9.x installed. Significantly faster for large files or large models. |

**Compute type** controls quantisation:

| Compute | Notes |
|---------|-------|
| **auto** | Recommended. Picks the best type for your device. |
| **int8** | Good for CPU. Smallest memory footprint. |
| **float16** | Good for CUDA. Faster than float32 on GPU. |
| **float32** | Full precision. Use if you see numerical issues. |

If you don't have an NVIDIA GPU, leave **Device** and **Compute** on **auto**. The app detects that CUDA is unavailable and falls back to CPU with int8 quantisation automatically.

---

## 9. Troubleshooting

### ffmpeg not found

**Symptom:** Error dialog at startup saying ffmpeg was not found.

**Fix:**
1. Open a terminal and run `ffmpeg -version`.
2. If it fails, ffmpeg is not on your PATH. Follow the installation steps in section 2.
3. If you just installed it, **restart your terminal** (or log out and back in) for PATH changes to take effect.

### Model download failure

**Symptom:** "Model load failed" error when clicking Transcribe.

**Fix:**
1. Check your internet connection — the first run for each model size requires a download.
2. If behind a proxy, set the `HTTPS_PROXY` environment variable.
3. Try a smaller model (e.g. `tiny` or `base`) to test connectivity.
4. Check free disk space — models can be up to 3 GB.
5. Check the log file (`audio_transcriber.log` in the project folder) for the full error.

### CUDA not detected / falls back to CPU

**Symptom:** Log shows "CUDA is not usable — falling back to CPU" even though you have an NVIDIA GPU.

**Fix:**
1. Verify the CUDA Toolkit 12.x is installed: check that `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin\cublas64_12.dll` exists.
2. Verify cuDNN 9.x is installed: check that `cudnn64_9.dll` (or similar) exists in the CUDA `bin` directory.
3. Check that `CUDA_PATH` is set: open a terminal and run `echo %CUDA_PATH%`. It should point to your CUDA installation (e.g. `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9`). If not set, add it manually:
   - Press `Win + R`, type `sysdm.cpl`, press Enter.
   - Go to **Advanced** → **Environment Variables**.
   - Under **User variables**, click **New**.
   - Variable name: `CUDA_PATH`, Value: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9` (adjust the version to match your install).
4. Verify CTranslate2 sees the GPU: `python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"` — should print a number > 0.
5. **Restart your IDE/terminal** after installing CUDA or changing environment variables. Environment changes only take effect in new processes.

### Empty transcript / no speech detected

**Symptom:** Output files are empty or log says "no speech detected".

**Possible causes:**
1. The audio file is silent, very short, or heavily compressed.
2. The language setting is wrong — if the audio is not Spanish, switch to **Auto-detect**.
3. Try a larger model (e.g. `medium` instead of `small`) for better sensitivity.
4. The audio format may be corrupt — try converting it manually: `ffmpeg -i input.ogg -ar 16000 output.wav` and transcribe the `.wav`.

### Permission denied when writing files

**Symptom:** "Write failed" error in the log.

**Fix:**
1. Make sure you have write permission to the folder where output files are saved.
2. If saving next to the source file, check that the source folder is not read-only (e.g. a USB drive or system folder).
3. Try setting a different **Output Folder** in the app to a location you control (e.g. your Desktop).
4. Close any program that might have the output file open.
