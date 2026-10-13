#!/usr/bin/env bash
# Serve the base model with the fine-tuned adapter, behind the same Chat Completions address
# the desk already speaks. Needs a Linux machine with an NVIDIA GPU.
#
# NOT RUN YET. Check each flag against `vllm serve --help` for the version you
# install. Flag names have changed between releases.
#
#   pip install vllm
#   bash finetune/serve_vllm.sh Qwen/Qwen2.5-1.5B-Instruct ./adapter
#
# What vLLM does that a plain generate() loop does not:
#   continuous batching   requests that arrive while others are mid-generation join the next
#                         step of the same batch, so the GPU is rarely idle
#   paged KV cache        each request's attention cache lives in fixed-size blocks, like
#                         pages of memory, so many requests of different lengths share the GPU
set -euo pipefail

BASE_MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
ADAPTER="${2:-./adapter}"

# --lora-modules name=path      serves the adapter under its own model name, beside the base model
# --enable-auto-tool-choice     lets the server turn the model's <tool_call> text into structured tool calls
# --tool-call-parser hermes     the parser for the <tool_call>{json}</tool_call> format this fine-tune writes
# --max-model-len               the longest prompt plus answer, in tokens. It sets how much GPU memory the cache needs
vllm serve "$BASE_MODEL" \
    --port 8001 \
    --enable-lora \
    --lora-modules "desk-tools=$ADAPTER" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90

# Then, from another terminal:
#   the tuned model   python finetune/evaluate.py --base-url http://localhost:8001/v1 --model desk-tools --data finetune/data/test.jsonl
#   the base model    python finetune/evaluate.py --base-url http://localhost:8001/v1 --model "$BASE_MODEL" --data finetune/data/test.jsonl
#   under load        python finetune/loadtest_endpoint.py --base-url http://localhost:8001/v1 --model desk-tools
#   the whole desk    DESK_LLM_BASE_URL=http://localhost:8001/v1 DESK_LLM_MODEL=desk-tools python eval/run_scenarios.py
