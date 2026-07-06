"""
LocalScribe pipeline: fully offline audio -> transcript -> summary.
Uses whisper.cpp for transcription and llama.cpp for summarization.
Both run 100% on-device via subprocess calls to the compiled binaries.
"""

import subprocess
import time
import json
import sys
import os

# ---- Paths (adjust if your build outputs land elsewhere) ----
WHISPER_BIN = "whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = "whisper.cpp/models/ggml-small.en.bin"
LLAMA_BIN = "llama.cpp/build/bin/llama-cli"

# Default LLM used by the app (swap path if you change quant/model)
DEFAULT_LLM_MODEL = "models/llama-3.2-3b-instruct-Q4_K_M.gguf"


def transcribe(audio_path: str, whisper_model: str = WHISPER_MODEL):
    """Run whisper.cpp on an audio file and return (transcript_text, elapsed_seconds)."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    start = time.time()
    result = subprocess.run(
        [WHISPER_BIN, "-m", whisper_model, "-f", audio_path, "-otxt", "-of", audio_path.rsplit(".", 1)[0]],
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        raise RuntimeError(f"whisper.cpp failed:\n{result.stderr}")

    txt_path = audio_path.rsplit(".", 1)[0] + ".txt"
    if not os.path.exists(txt_path):
        raise RuntimeError(f"Expected transcript not found at {txt_path}. stderr:\n{result.stderr}")

    with open(txt_path, "r") as f:
        transcript = f.read().strip()

    return transcript, elapsed


def summarize(transcript: str, model_path: str = DEFAULT_LLM_MODEL, max_tokens: int = 300):
    """Run llama.cpp on a transcript and return (summary_text, elapsed_seconds)."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"LLM model not found: {model_path}")

    prompt = (
        "Summarize the following transcript into clear key points and action items. "
        "Be concise and use bullet points.\n\n"
        f"Transcript:\n{transcript}\n\nSummary:"
    )

    start = time.time()
    result = subprocess.run(
        [LLAMA_BIN, "-m", model_path, "-p", prompt, "-n", str(max_tokens), "--no-display-prompt"],
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        raise RuntimeError(f"llama.cpp failed:\n{result.stderr}")

    return result.stdout.strip(), elapsed


def run_pipeline(audio_path: str, llm_model: str = DEFAULT_LLM_MODEL):
    """Full pipeline: audio -> transcript -> summary, with timing for both stages."""
    transcript, t_transcribe = transcribe(audio_path)
    summary, t_summarize = summarize(transcript, model_path=llm_model)

    return {
        "transcript": transcript,
        "summary": summary,
        "transcribe_seconds": round(t_transcribe, 2),
        "summarize_seconds": round(t_summarize, 2),
        "total_seconds": round(t_transcribe + t_summarize, 2),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <audio_path> [llm_model_path]")
        sys.exit(1)

    audio_arg = sys.argv[1]
    model_arg = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_LLM_MODEL

    output = run_pipeline(audio_arg, model_arg)
    print(json.dumps(output, indent=2))
