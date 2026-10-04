# AutoShorts 11.0

> **Automated AI-Powered Short-Form Video Repurposing & Multi-Speaker Vertical Cropping**

AutoShorts 11.0 is an enterprise-grade desktop application that transforms long-form horizontal videos (podcasts, interviews, streams, YouTube videos) into high-impact, vertically framed (9:16 portrait) short-form clips with intelligent active-speaker tracking, automated silence/pacing editing, phrase-paired kinetic captions, and multi-LLM viral moment scoring.

---

## 🌟 Key Architecture & Capabilities

### 1. Multi-Branch Intelligent Video Framing (DualFrame)
- **Active Speaker Tracking**: Combines YOLOv11 person detection, BoT-SORT identity tracking, and audio-visual speaker fusion to center on active speakers in real-time.
- **DualFrame Adaptive Splitting**: Automatically splits between dynamic solo speaker portrait crops and multi-speaker split-screen layouts depending on conversational dynamics and participant proximity.
- **Cinematic Smoothing**: Applies Kalman filtering and hysteresis damping to prevent erratic camera jerks and jarring cuts.

### 2. Audio Intelligence & Smart Pacing
- **Pause Intelligence**: LightGBM 6-class pause classifier (`breath_pause`, `waiting_pause`, `sentence_pause`, `normal_word_gap`, `nonspeech_unknown`, `speaker_transition`) that prunes disfluencies while preserving comedic and dramatic timing.
- **PANNs Acoustic Reaction Extraction**: Analyzes laughter, applause, music transitions, and emotional acoustic cues.
- **Strict Authority Invariants**: Deterministic keep-rules always safeguard spoken speech buffers and audio fidelity.

### 3. T7 Phrase-Paired Dynamic Captions
- **Word-Level Synchronization**: Sub-frame accurate word timing aligned with phonetic boundaries.
- **Three-Role Styling Engine**: Distinctive typography and color palettes for primary speech, secondary decorative remarks, and high-impact emphasis words.
- **Direct ASS Subtitle Rendering**: Uses native libass / FFmpeg subtitles filters for high-fidelity typography without web-layer jitter.

### 4. Viral Candidate Discovery
- **REZE Window Scoring**: NVIDIA NIM (`google/diffusiongemma-26b-a4b-it`) reasoning engine that scores timeline windows across narrative hook, insight density, and emotional punch.
- **DeepSeek & Claude Pipelines**: Multi-LLM timestamp generation with automatic fail-soft fallback.
- **Advisory VLM**: Visual-Language Model support (NVIDIA Nemotron 3 Nano Omni) for multimodal visual scene quality assessment.

---

## 📂 Repository Structure

```
.
├── autoshorts/                     # Main Desktop Application (Tauri 2 + React + Rust)
│   ├── src/                        # React 18 + TypeScript + Vite + Tailwind UI
│   │   ├── components/             # Video preview, timeline editor, candidate inspector
│   │   └── main.tsx                # Application shell and UI routing
│   ├── src-tauri/                  # Rust Backend Engine
│   │   ├── Cargo.toml              # Rust crate dependencies
│   │   ├── src/                    # Core modules (media, framing, pacing, captions, llm)
│   │   ├── scripts/                # Python sidecars (speaker tracking, LightGBM models)
│   │   └── tests/                  # Rust integration test suites
│   ├── package.json                # Frontend dependencies and build scripts
│   └── .env.example                # Application environment variable template
├── docs/                           # Architecture, reports, development plans & archives
│   ├── architecture/               # System architecture & core ML research reports
│   ├── development/                # Implementation plans and test infrastructure docs
│   ├── reports/                    # Phase execution, audit, and validation reports
│   ├── license_audit/              # Compliance audits for ML models and third-party tools
│   ├── superpowers/                # Detailed engineering specifications & plans
│   └── archive/                    # Archived project notes and historical requests
├── tests/                          # Standalone integration and regression test suites
│   ├── test_applied_features_suite.py
│   ├── test_audio_intelligence_suite.py
│   ├── test_caption_qa_suite.py
│   ├── test_hook_closure_suite.py
│   ├── test_hook_ending_optimization_suite.py
│   ├── test_smart_pacing_suite.py
│   ├── test_smart_pacing_2_suite.py
│   └── test_zero.wav
├── Fontfabric-Matt-Trial/          # Typography assets for ASS caption engine
├── requirements.txt                # Root Python dependencies for DSP and ML sidecars
├── run_test.bat                    # Windows batch runner for test suites
├── .env.example                    # Root environment configuration template
└── README.md                       # Repository documentation
```

---

## 🛠️ Prerequisites

To run and build AutoShorts 11.0, your development environment requires:

1. **Operating System**: Windows 10/11 (64-bit), macOS 12+, or modern Linux (Ubuntu 22.04+).
2. **FFmpeg & FFprobe**: Must be installed and accessible on system `PATH`.
   - *Windows (winget)*: `winget install Gyan.FFmpeg`
   - *macOS (Homebrew)*: `brew install ffmpeg`
   - *Linux*: `sudo apt update && sudo apt install ffmpeg`
   - *Verify*: `ffmpeg -version` (ensure `libass` and `libx264` support is compiled in).
3. **Node.js**: Node.js 18.x or 20.x LTS with `npm`.
4. **Rust Toolchain**: Rust 1.78+ (via `rustup`).
   - Run `rustup update stable`
5. **Python**: Python 3.10, 3.11, or 3.12 (64-bit).
6. **C++ Build Tools**: 
   - *Windows*: Visual Studio C++ Build Tools (Desktop development with C++).
   - *Linux*: `build-essential libwebkit2gtk-4.1-dev libappindicator3-dev`

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Nakshtra07/Shortform.git
cd Shortform
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Edit `.env` and fill in your API credentials:
```env
# Cloud Transcription (Fast, accurate ASR)
DEEPGRAM_API_KEY=your_deepgram_api_key_here

# LLM Providers for Candidate Discovery
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Optional: NVIDIA NIM API Key (For REZE DiffusionGemma & Nemotron VLM)
NVIDIA_API_KEY=your_nvidia_api_key_here
```

### 3. Set Up Python Environment
Create a virtual environment and install the required dependencies:
```bash
# In the repository root:
python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate

# Activate on macOS/Linux:
source .venv/bin/activate

# Install dependencies:
pip install -r requirements.txt
```

### 4. Install Frontend Dependencies
```bash
cd autoshorts
npm install
```

### 5. Launch Development Server
To launch the live application with hot-reloading for both the React frontend and Rust backend:
```bash
npm run tauri:dev
```

### 6. Build Standalone Installer
To package the native desktop application installer (`.msi` / `.exe` on Windows, `.dmg` on macOS, `.deb` on Linux):
```bash
npm run tauri:build
```
Packaged binaries will be located in `autoshorts/src-tauri/target/release/bundle/`.

---

## 🧪 Verification & Testing

AutoShorts contains extensive test suites across both Rust and Python:

### Rust Test Suite
```bash
cd autoshorts/src-tauri
cargo test --lib -j 2
cargo test --test pause_intel_suite -j 2
cargo test --test t7_prosody_suite -j 2
```

### Python DSP & Intelligence Test Suite
```bash
# From the root directory with .venv activated:
python -m unittest discover -s tests -p "test_*.py"

# Or run individual suites directly:
python tests/test_applied_features_suite.py
python tests/test_audio_intelligence_suite.py
python tests/test_smart_pacing_suite.py
python tests/test_smart_pacing_2_suite.py
python tests/test_hook_closure_suite.py
python tests/test_hook_ending_optimization_suite.py
```

---

## 📜 License & Compliance

- **Application Code**: Licensed under the MIT License.
- **Third-Party Model Audits**: Full open-source compliance documentation and license audits for underlying models (YOLOv11, PANNs, LightGBM, Whisper) are documented in [`docs/license_audit/`](docs/license_audit/).
