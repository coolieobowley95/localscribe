#!/bin/bash
# run_quant_comparison.sh
# Benchmarks the same LLM at two quantization levels (Q8_0 vs Q4_K_M)
# and records size, memory, and speed deltas for the LocalScribe README.
#
# Usage:
#   bash benchmarks/run_quant_comparison.sh
#
# Requires: models/llama-3.2-3b-instruct-Q8_0.gguf
#           models/llama-3.2-3b-instruct-Q4_K_M.gguf
# (download both quantizations of the same base model from Hugging Face
#  and place them in ./models/ before running)

set -e

LLAMA_BIN="llama.cpp/build/bin/llama-cli"
PROMPT="Summarize the following meeting notes into key points and action items: The team discussed Q3 roadmap priorities, agreed to delay the mobile release by two weeks, and assigned follow-up tasks to engineering and design."
N_TOKENS=200
OUT_FILE="benchmarks/quantization_results.md"

MODELS=(
  "Q8_0:models/llama-3.2-3b-instruct-Q8_0.gguf"
  "Q4_K_M:models/llama-3.2-3b-instruct-Q4_K_M.gguf"
)

echo "# Quantization Benchmark Results" > "$OUT_FILE"
echo "" >> "$OUT_FILE"
echo "| Quant | Size on disk | Peak memory | Wall time (${N_TOKENS} tokens) |" >> "$OUT_FILE"
echo "|---|---|---|---|" >> "$OUT_FILE"

for entry in "${MODELS[@]}"; do
  LABEL="${entry%%:*}"
  MODEL_PATH="${entry##*:}"

  if [ ! -f "$MODEL_PATH" ]; then
    echo "WARNING: $MODEL_PATH not found, skipping $LABEL"
    continue
  fi

  SIZE=$(du -h "$MODEL_PATH" | cut -f1)

  echo ""
  echo "=== Running $LABEL ($MODEL_PATH) ==="

  # /usr/bin/time -l gives peak memory on macOS (BSD time).
  # On Linux, swap to: /usr/bin/time -v ... and parse "Maximum resident set size"
  START=$(date +%s.%N)
  /usr/bin/time -l "$LLAMA_BIN" -m "$MODEL_PATH" -p "$PROMPT" -n "$N_TOKENS" > "/tmp/${LABEL}_output.txt" 2> "/tmp/${LABEL}_time.txt"
  END=$(date +%s.%N)

  ELAPSED=$(echo "$END - $START" | bc)

  # Extract peak memory (macOS: "maximum resident set size" in bytes)
  PEAK_MEM_BYTES=$(grep "maximum resident set size" "/tmp/${LABEL}_time.txt" | awk '{print $1}')
  if [ -n "$PEAK_MEM_BYTES" ]; then
    PEAK_MEM_MB=$(echo "$PEAK_MEM_BYTES / 1024 / 1024" | bc)
    PEAK_MEM="${PEAK_MEM_MB} MB"
  else
    PEAK_MEM="see /tmp/${LABEL}_time.txt"
  fi

  echo "| $LABEL | $SIZE | $PEAK_MEM | ${ELAPSED}s |" >> "$OUT_FILE"
  echo "Done: $LABEL -> size=$SIZE, peak_mem=$PEAK_MEM, time=${ELAPSED}s"
done

echo ""
echo "Results written to $OUT_FILE"
cat "$OUT_FILE"
