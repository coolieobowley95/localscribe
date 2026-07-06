"""
LocalScribe UI: a minimal local Gradio app to demo the offline
transcription + summarization pipeline. Meant to be run with
Wi-Fi disabled to prove the "fully offline" claim on camera.
"""

import gradio as gr
from pipeline import run_pipeline, DEFAULT_LLM_MODEL


def process(audio_path):
    if audio_path is None:
        return "No audio provided.", "", ""

    try:
        result = run_pipeline(audio_path, DEFAULT_LLM_MODEL)
    except Exception as e:
        return f"Error: {e}", "", ""

    timing = (
        f"Transcription: {result['transcribe_seconds']}s | "
        f"Summarization: {result['summarize_seconds']}s | "
        f"Total: {result['total_seconds']}s"
    )
    return result["transcript"], result["summary"], timing


with gr.Blocks(title="LocalScribe — 100% Offline Voice Notes") as demo:
    gr.Markdown(
        "# LocalScribe\n"
        "Fully offline voice-note transcription and summarization. "
        "Try disabling Wi-Fi before running — it still works."
    )

    audio_input = gr.Audio(type="filepath", label="Upload or record audio")
    run_button = gr.Button("Process", variant="primary")

    transcript_output = gr.Textbox(label="Transcript", lines=8)
    summary_output = gr.Textbox(label="Summary", lines=8)
    timing_output = gr.Textbox(label="Timing")

    run_button.click(
        fn=process,
        inputs=audio_input,
        outputs=[transcript_output, summary_output, timing_output],
    )

if __name__ == "__main__":
    demo.launch()
