# Real-Time Audio Processing & Transcription Engine

An enterprise-grade, modular audio processing pipeline designed for real-time speech capturing, enhancement, Voice Activity Detection (VAD), and automated transcription. This ecosystem provides a robust foundation for building low-latency voice assistants, automated meeting scribes, or local intelligence workflows.

## 🚀 System Architecture & Module Overview

The system is fully decoupled and optimized for modular deployment. Below is the layout and functional purpose of each core component:

```
Final Project/
├── .vscode/               # Workspace specific development configurations
├── models/                # Directory for local model checkpoints (VAD, ASR, etc.)
├── aec.py                 # Acoustic Echo Cancellation (AEC) module
├── agc.py                 # Automatic Gain Control (AGC) module
├── audio_engine.py        # Core I/O engine for managing audio streams and hardware interfaces
├── audio_processor.py     # High-level pipeline orchestration (linking DSP modules)
├── vad.py                 # Voice Activity Detection (VAD) filtering
├── transcriber.py         # Speech-to-Text inference layer (ASR engine)
├── postprocess.py         # Transcript cleanup, text formatting, and normalization
├── config.py              # Application-wide hyperparameter and environment settings
├── logger.py              # Centralized runtime logging and telemetry
├── utils.py               # Auxiliary helper functions (buffers, math, conversions)
├── main.py                # Main orchestration script and application entry point
└── requirements.txt       # Project dependency manifest
```

### Core Components Deep-Dive

*   **`audio_engine.py`**: Interacts directly with hardware/virtual sound interfaces to capture low-latency microphone frames or stream inputs. Handles buffer queues cleanly to prevent underruns or drops.
*   **Digital Signal Processing (`aec.py` & `agc.py`)**: 
    *   **AEC**: Removes loudspeaker feedback from the captured audio channel when simultaneous playing and recording occur.
    *   **AGC**: Dynamic range compression and normalization ensuring consistent audio volume amplitudes across loud/quiet environments before running inferences.
*   **`vad.py`**: Filters ambient noise and silent frames, slicing long continuous audio inputs into discrete speech segments to drastically reduce downstream ASR compute load.
*   **`transcriber.py`**: Responsible for model management and inference. Translates audio frames into text representations.
*   **`postprocess.py`**: Sanitizes the resulting raw transcripts (e.g., adding punctuation, filtering filler words, capitalizing names, or structure mapping).

---

## 🛠️ Getting Started

### Prerequisites
- Python 3.13+
- Sound drivers properly configured locally (e.g., PortAudio if interacting with microphone arrays).

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/audio-transcription-engine.git
   cd audio-transcription-engine/Final\ Project
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Model Provisioning:**
   Place required model artifacts or weights inside the `models/` directory according to your backend settings specified in `config.py`.

---

## 💻 Usage

To launch the complete real-time processing and transcription engine pipeline, run:

```bash
python main.py
```

### Performance & Configuration Tunings
You can tune hardware buffer sizing, DSP frame windows, VAD confidence thresholds, and model execution devices (such as local `cuda` / `cpu`) directly inside `config.py`.

---

## 📊 Features & Engineering Highlights

- **Streaming Architecture**: Fully multi-threaded frame buffers ensuring independent processing of hardware recording and backend ASR inference.
- **Robust DSP Pipeline**: Integrated Acoustic Echo Cancellation and Automatic Gain Control to ensure high quality input in non-ideal sonic environments.
- **Intelligent VAD Filtering**: Eliminates redundant compute cost by keeping the neural networks idle unless human speech is actively detected.
- **Modular Post-Processing**: Structured to easily attach custom downstream agents or LLM parsing pipelines right after transcripts are finalized.

---

## 📄 License
This project is licensed under the MIT License - see your local repository details for specifics.
README.md
Displaying README.md.
