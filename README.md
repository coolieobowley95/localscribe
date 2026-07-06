# LocalScribe — 100% Offline Voice Note Transcription & Summarization

**Arm Create: AI Optimization Challenge 2026 — Mobile AI Track**

LocalScribe turns a voice memo or meeting recording into a clean, structured summary — entirely on-device. No cloud APIs, no network calls, no data leaving your laptop. Built and optimized to run efficiently on Arm-powered devices (Apple Silicon / Snapdragon / Arm Windows).

---

## Project Overview

Most voice-note and meeting-summarization tools require sending your audio to a cloud API — which means privacy trade-offs, latency, and a dependency on network connectivity. LocalScribe removes all of that: transcription and summarization both run locally using Arm-optimized inference engines (`whisper.cpp` and `llama.cpp`), so the entire pipeline works with Wi-Fi turned off.

**What makes it interesting:**
- Zero external API calls — verified by running the full demo with networking disabled.
- Real Arm-specific optimization: both engines use NEON/quantization kernels tuned for Arm CPUs, not generic x86 builds.
- Quantifiable optimization story: we benchmark the same LLM at two quantization levels (Q8_0 vs Q4_K_M) and report the size, memory, and speed deltas — see [Benchmarks](#benchmarks) below.
- Directly reusable: the pipeline script and benchmarking harness can be dropped into any other on-device Arm AI project.

**Why it should win:** it directly answers the Mobile AI track's core ask — privacy, latency, battery efficiency, and local AI experience — with a working, measurable, demoable product rather than a proof-of-concept.

---

## Functionality / Output

**Input:** an audio file (`.wav`, `.m4a`, `.mp3`) — a voice memo, meeting recording, or lecture.

**Pipeline:**
1. `whisper.cpp` transcribes the audio to text, fully on-device.
2. `llama.cpp` takes the transcript and produces a structured summary (key points + action items), fully on-device.

**Output:** a transcript file and a summary (markdown), plus timing metrics for each stage, displayed in a simple local UI.

---

## Models Used

| Component | Model | Quantization | Approx. Size |
|---|---|---|---|
| Speech-to-text | Whisper `[base.en / small.en / medium.en]` | ggml | `[fill in]` |
| Summarization | `[Qwen2.5-1.5B-Instruct / Llama-3.2-3B-Instruct / Qwen2.5-7B-Instruct]` | Q4_K_M (compared against Q8_0) | `[fill in]` |

Model selection was based on available RAM — see the project writeup for the sizing table used to choose these.

---

## Setup Instructions

Tested on: `[Apple Silicon Mac / Snapdragon Windows laptop — fill in your device]`

### 1. Clone this repository
```bash
git clone https://github.com/<your-username>/localscribe
cd localscribe
```

### 2. Build the inference engines
```bash
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
cmake -B build -DGGML_METAL=1   # remove -DGGML_METAL=1 on non-Apple Arm devices
cmake --build build --config Release
cd ..

git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_METAL=1
cmake --build build --config Release
cd ..
```

### 3. Download models
```bash
bash whisper.cpp/models/download-ggml-model.sh base.en

# Download your chosen instruct model in GGUF format from Hugging Face
# and place it in ./models/
```

### 4. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the app
```bash
python src/app.py
```
This launches a local Gradio UI at `http://127.0.0.1:7860`. Upload or record an audio clip and the transcript + summary will appear, along with timing for each stage.

### 6. Verify it's fully offline
Turn off Wi-Fi, then repeat step 5 — the app will still function identically.

---

## Benchmarks

Same audio input, same LLM, two quantization levels:

| Metric | Q8_0 | Q4_K_M | Change |
|---|---|---|---|
| Model size on disk | `[fill in]` | `[fill in]` | `[fill in]` |
| Peak memory usage | `[fill in]` | `[fill in]` | `[fill in]` |
| Tokens/sec | `[fill in]` | `[fill in]` | `[fill in]` |
| Time to first token | `[fill in]` | `[fill in]` | `[fill in]` |
| Summary quality (manual review) | `[fill in]` | `[fill in]` | `[fill in]` |

Benchmarking methodology and raw output are in [`/benchmarks`](./benchmarks). Numbers cross-checked with [Arm Performix](https://www.arm.com) where applicable.

To reproduce:
```bash
bash benchmarks/run_quant_comparison.sh
```

---

## Repository Structure

```
localscribe/
├── src/
│   ├── pipeline.py       # transcription + summarization pipeline
│   └── app.py            # Gradio UI
├── benchmarks/
│   ├── run_quant_comparison.sh
│   └── quantization_results.md
├── docs/
│   └── benchmark_chart.png
├── models/               # (not committed — download instructions above)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Demo Video

`[Link to YouTube/Vimeo demo video — under 3 minutes, showing the app running with Wi-Fi disabled and the benchmark results on screen]`

---

## License

Licensed under the MIT License — see [LICENSE](./LICENSE) for details.

---

## Built With

- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) — on-device speech-to-text, Arm-optimized
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — on-device LLM inference, Arm-optimized
- [Gradio](https://www.gradio.app/) — local UI
- Arm Performix — benchmarking
