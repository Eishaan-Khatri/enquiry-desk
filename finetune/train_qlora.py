"""Step 2. Fine-tune a small model to make tool calls, with QLoRA. Runs on a GPU.

NOT RUN YET. My laptop has no NVIDIA GPU. The script is
written against the documented APIs of transformers, peft and bitsandbytes. Run it on Kaggle
or Colab (see finetune/README.md) and expect to fix small things on the first run.

    python finetune/train_qlora.py --model Qwen/Qwen2.5-1.5B-Instruct --train data/train.jsonl --out adapter/

What QLoRA is, in three lines:
  Q     the base model's weights are loaded in 4 bits, not 16, so it fits in a small GPU.
  LoRA  the base weights are frozen. Beside each chosen weight matrix W, two small matrices
        A and B are trained, and the layer computes W x + B A x. A and B together are well
        under 1% of the model.
  So training a multi-billion-parameter model needs about as much memory as running it.

This file is self-contained on purpose. It imports nothing from the desk, so the GPU machine
needs only: pip install torch transformers peft bitsandbytes accelerate datasets mlflow
"""

from __future__ import annotations

import argparse
import json

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

SYSTEM = (
    "You can call the tools described below. To call one, reply with a JSON object on its own, "
    'like {"name": "tool_name", "arguments": {...}}, wrapped in <tool_call> tags. Use one pair of tags per call. '
    "If no tool fits the request, answer in plain words and call nothing.\n\nTools:\n"
)


def to_chat(example: dict) -> list[dict]:
    """The same format as enquiry_desk.finetune.data.to_chat. Keep the two in step."""
    calls = "\n".join(f"<tool_call>\n{json.dumps(a, ensure_ascii=False)}\n</tool_call>" for a in example["answers"])
    return [
        {"role": "system", "content": SYSTEM + json.dumps(example["tools"], ensure_ascii=False)},
        {"role": "user", "content": example["query"]},
        {"role": "assistant", "content": calls or "No tool fits this request."},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train", required=True)
    parser.add_argument("--out", default="adapter")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16, help="the width of A and B. Higher learns more and costs more")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--limit", type=int, help="train on the first N rows, for a trial run")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(args.train, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows = rows[: args.limit] if args.limit else rows

    def encode(example: dict) -> dict:
        messages = to_chat(example)
        # The prompt is everything up to where the assistant starts. The loss is computed only
        # on what comes after it. Otherwise the model is also trained to recite tool schemas,
        # which wastes its capacity on text it will always be given.
        prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        full = prompt + messages[-1]["content"] + tokenizer.eos_token
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        ids = tokenizer(full, add_special_tokens=False, truncation=True, max_length=args.max_length)["input_ids"]
        labels = [-100] * min(len(prompt_ids), len(ids)) + ids[len(prompt_ids):]  # -100 means "do not score this token"
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}

    dataset = Dataset.from_list([{"example": json.dumps(row)} for row in rows]).map(
        lambda item: encode(json.loads(item["example"])), remove_columns=["example"]
    )
    # An example cut off before its answer began has nothing left to learn from.
    dataset = dataset.filter(lambda item: any(label != -100 for label in item["labels"]))

    def collate(batch: list[dict]) -> dict:
        longest = max(len(item["input_ids"]) for item in batch)
        pad = tokenizer.pad_token_id

        def padded(key: str, fill: int) -> torch.Tensor:
            return torch.tensor([item[key] + [fill] * (longest - len(item[key])) for item in batch])

        return {"input_ids": padded("input_ids", pad), "attention_mask": padded("attention_mask", 0), "labels": padded("labels", -100)}

    quantisation = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # a 4-bit code fitted to the bell shape that weights have
        bnb_4bit_compute_dtype=torch.bfloat16,  # weights are stored in 4 bits and computed in 16
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quantisation, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank,
            lora_alpha=2 * args.rank,  # the update is scaled by alpha / r, so this keeps the scale at 2
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        data_collator=collate,
        args=TrainingArguments(
            output_dir=args.out,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=2,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            logging_steps=20,
            save_strategy="epoch",
            bf16=True,
            seed=args.seed,
            report_to=["mlflow"],  # every run's settings and loss curve are kept, so runs can be compared
            run_name=f"qlora-{args.model.split('/')[-1]}-r{args.rank}-seed{args.seed}",
        ),
    )
    trainer.train()
    model.save_pretrained(args.out)  # only A and B are saved: a few tens of megabytes
    tokenizer.save_pretrained(args.out)
    print(f"Adapter saved to {args.out}")


if __name__ == "__main__":
    main()
