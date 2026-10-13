"""Write finetune/kaggle_notebook.ipynb. Edit the cells here, then run: uv run python finetune/make_notebook.py"""

from __future__ import annotations

import json
from pathlib import Path

CELLS = [
    ("markdown", """# Fine-tune a small model for the desk's tool calls

**Not run yet.** Written for a Kaggle or Colab notebook with a T4 GPU. Run the cells in order and expect to fix small things on the first pass. The reasoning for each step is in `finetune/README.md`.

Turn the GPU on first: *Settings > Accelerator > GPU T4*."""),
    ("code", """# 1. The project and the libraries. Replace the URL with your own repository.
!git clone https://github.com/YOUR_USER/enquiry-desk.git
%cd enquiry-desk
!pip install -q -e . torch transformers peft bitsandbytes accelerate datasets mlflow vllm"""),
    ("code", """# 2. The public data. Check the licence and access terms of whichever set you use.
#    This cell expects a JSONL file in the xLAM format: query, tools, answers.
from datasets import load_dataset
rows = load_dataset("Salesforce/xlam-function-calling-60k", split="train")   # needs you to accept its terms on the Hub
rows.to_json("xlam.jsonl")"""),
    ("code", """# 3. Split by tool, drop leaks, and READ THE REPORT before going on.
!python finetune/prepare_data.py --public xlam.jsonl --out finetune/data
!cat finetune/data/leak_report.json"""),
    ("code", """# 4. Baseline: serve the untuned model and score it on the WHOLE test file.
import subprocess, time
BASE = "Qwen/Qwen2.5-1.5B-Instruct"
server = subprocess.Popen(["vllm", "serve", BASE, "--port", "8001", "--max-model-len", "8192"])
time.sleep(180)   # the model takes a few minutes to load
!python finetune/evaluate.py --data finetune/data/test.jsonl --base-url http://localhost:8001/v1 --model {BASE} --concurrency 16 --out finetune/data/scores_base.json
server.terminate()"""),
    ("code", """# 5. Train. Try --limit 500 first to see that it runs, then remove it.
!python finetune/train_qlora.py --model {BASE} --train finetune/data/train.jsonl --out adapter --limit 500"""),
    ("code", """# 6. Serve the base model with the adapter, and score it with the same command as the baseline.
server = subprocess.Popen(["bash", "finetune/serve_vllm.sh", BASE, "./adapter"])
time.sleep(180)
!python finetune/evaluate.py --data finetune/data/test.jsonl --base-url http://localhost:8001/v1 --model desk-tools --concurrency 16 --out finetune/data/scores_tuned.json"""),
    ("code", """# 7. Speed under load. Throughput should rise with the number of callers.
!python finetune/loadtest_endpoint.py --base-url http://localhost:8001/v1 --model desk-tools --levels 1,4,16,32"""),
    ("code", """# 8. Compare. If the tuned model is no better than the base, that is the result to report.
import json
for name in ("base", "tuned"):
    s = json.load(open(f"finetune/data/scores_{name}.json"))["summary"]
    print(name, {k: (v["rate"], v["interval_95"]) for k, v in s.items() if isinstance(v, dict)})"""),
    ("markdown", """## Before you quote a number

- It must come from the whole test file, not `--limit`.
- `test_queries_also_in_train` in the leak report must be 0.
- Read 100 failures by hand and sort them into kinds.
- Then point the desk at the tuned model and run `eval/run_scenarios.py`. Single calls are not conversations."""),
]


def main() -> None:
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
        "cells": [
            {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
            | ({"outputs": [], "execution_count": None} if kind == "code" else {})
            for kind, text in CELLS
        ],
    }
    out = Path(__file__).parent / "kaggle_notebook.ipynb"
    out.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Wrote {out} with {len(CELLS)} cells.")


if __name__ == "__main__":
    main()
