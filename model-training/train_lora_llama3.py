import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def infer_text(example: Dict[str, Any]) -> str:
    if "text" in example and example["text"] is not None:
        return str(example["text"])

    prompt = example.get("prompt")
    response = example.get("response")
    instruction = example.get("instruction")
    input_ = example.get("input")
    output = example.get("output")

    if prompt is not None and response is not None:
        return f"{prompt}\n{response}"
    if instruction is not None and output is not None:
        if input_:
            return f"### Instruction\n{instruction}\n\n### Input\n{input_}\n\n### Response\n{output}"
        return f"### Instruction\n{instruction}\n\n### Response\n{output}"

    return json.dumps(example, ensure_ascii=False)


def build_tokenize_fn(tokenizer, max_seq_length: int):
    def tokenize_batch(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        texts = [infer_text(ex) for ex in batch["__raw__"]]
        out = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_attention_mask=True,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    return tokenize_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="dataset.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/lora-llama3-8b")

    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated list of module names to target with LoRA.",
    )

    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load base model in 4-bit via bitsandbytes (recommended for 8B on limited VRAM).",
    )
    parser.add_argument(
        "--use_bf16",
        action="store_true",
        help="Prefer bf16 (if supported). Otherwise fp16 is used on CUDA.",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="paged_adamw_8bit",
        help="Optimizer name for TrainingArguments (e.g., paged_adamw_8bit, adamw_torch).",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {"torch_dtype": torch.bfloat16 if args.use_bf16 else None}
    if args.load_in_4bit:
        model_kwargs.update(
            {
                "load_in_4bit": True,
                "device_map": "auto",
            }
        )
    else:
        if torch.cuda.is_available():
            model_kwargs.update({"device_map": "auto"})

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_targets = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_targets,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    dataset = load_dataset("json", data_files={"train": args.dataset_path})
    train_ds = dataset["train"]
    train_ds = train_ds.map(lambda ex: {"__raw__": ex}, remove_columns=train_ds.column_names)
    tokenize_fn = build_tokenize_fn(tokenizer, args.max_seq_length)
    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["__raw__"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    use_cuda = torch.cuda.is_available()
    use_bf16 = bool(args.use_bf16 and use_cuda and torch.cuda.is_bf16_supported())
    use_fp16 = bool(use_cuda and not use_bf16)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_strategy="steps",
        evaluation_strategy="no",
        report_to="none",
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        optim=args.optim,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
