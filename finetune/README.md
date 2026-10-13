# Fine-tuning a small model for the desk's tool calls

**The question.** Can a small open model make this desk's tool calls, and what is lost against a hosted model?

**Status.** The data, scoring and evaluation code is tested and has been run. The training script and the vLLM commands have **not been run**: the laptop this was built on has no NVIDIA GPU. They run on Kaggle or Colab, which need your own login.

| File | What it does | Run so far? |
|---|---|---|
| `prepare_data.py` | Builds train and test files, split by tool, with a leak report | Yes, for the desk's generated examples |
| `train_qlora.py` | QLoRA fine-tune with MLflow logging | **No** |
| `evaluate.py` | Scores a model on every example of a test file | Yes, on a local model |
| `serve_vllm.sh` | Serves the base model with the adapter | **No** |
| `loadtest_endpoint.py` | Throughput and first-token time at rising load | Yes, on a local model |
| `../src/enquiry_desk/finetune/` | The split, the leak check and the scoring, with tests | Yes, 12 tests |

## One result so far

`qwen2.5:3b`, untuned, on 60 generated desk examples, every example scored (13 Oct 2026):

| Measure | Rate | 95% interval |
|---|---|---|
| Output could be parsed | 60 of 60 | |
| Named the right tools | 98% | 95% to 100% |
| **Every argument right** | **68%** | 57% to 80% |
| Called nothing when nothing was needed | 9 of 9 | |

So the small model nearly always picks the right tool and gets an argument wrong in about a third of calls. That gap, between the right tool and the right arguments, is what a fine-tune should close. The examples are generated from templates, so read the figures as a baseline to beat, not as how the model does on real callers.

## The steps, on Kaggle or Colab

1. **Data.** Get a public function-calling set in the xLAM format (`query`, `tools`, `answers`) as JSONL. Check its licence and access terms first. Then:

   ```bash
   python finetune/prepare_data.py --public xlam.jsonl --out finetune/data
   ```

   Read `leak_report.json` before going on. `test_queries_also_in_train` must be 0.

2. **Baseline.** Serve the untuned model and score it on the whole test file. This is the number the fine-tune has to beat.

3. **Train.**

   ```bash
   python finetune/train_qlora.py --model Qwen/Qwen2.5-1.5B-Instruct --train finetune/data/train.jsonl --out adapter
   ```

   Try `--limit 500` first. A full run on a free T4 is a few hours.

4. **Serve** with `serve_vllm.sh`, then **score** with the same command as the baseline:

   ```bash
   python finetune/evaluate.py --data finetune/data/test.jsonl --base-url http://localhost:8001/v1 --model desk-tools
   ```

5. **Load test.**

   ```bash
   python finetune/loadtest_endpoint.py --base-url http://localhost:8001/v1 --model desk-tools
   ```

6. **The test that matters.** Point the desk at the tuned model (`DESK_LLM_BASE_URL`, `DESK_LLM_MODEL`) and run `eval/run_scenarios.py`. A model can score well on single calls and still fail a conversation.

## What to report

- The table above for base, tuned and one hosted model, on **the whole test file**.
- The leak report.
- 100 failures read by hand and sorted into kinds.
- Tokens per second and first-token time at 1, 4, 16 and 32 callers.
- If the tuned model is no better than the base, say so. That is a result.

## Not checked yet

- Whether `train_qlora.py` runs as written.
- Whether the vLLM flags in `serve_vllm.sh` match the version you install. Flag names have changed between releases.
- Any number for a tuned model. None exists.
