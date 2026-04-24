import argparse
import os

import wandb
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from .config import Config, load_config
from .data import prepare


def build_unsloth(cfg: Config):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.train.max_seq_length,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.train.lora_r,
        lora_alpha=cfg.train.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    return model, tokenizer


def build_hf(cfg: Config):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    lora = LoraConfig(
        r=cfg.train.lora_r,
        lora_alpha=cfg.train.lora_alpha,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora), tokenizer


def tokenize(ds, tokenizer, max_length: int):
    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    return ds.map(_tok, batched=True, remove_columns=["text"])


def run(profile: str):
    cfg = load_config(profile)

    os.environ["WANDB_MODE"] = cfg.wandb.mode
    wandb.init(
        project=cfg.wandb.project,
        config=cfg.model_dump(mode="json"),
        mode=cfg.wandb.mode,
        name=f"{profile}-{cfg.model_name.split('/')[-1]}",
    )

    ds = prepare(cfg.dataset)
    model, tokenizer = build_unsloth(cfg) if cfg.train.use_unsloth else build_hf(cfg)
    ds = tokenize(ds, tokenizer, cfg.train.max_seq_length)

    cfg.train.output_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(cfg.train.output_dir),
        max_steps=cfg.train.max_steps,
        per_device_train_batch_size=cfg.train.batch_size,
        learning_rate=cfg.train.learning_rate,
        logging_steps=1,
        save_strategy="no",
        report_to=["wandb"] if cfg.wandb.mode != "disabled" else [],
    )
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)
    trainer.train()

    adapter_dir = cfg.train.output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    if cfg.wandb.mode != "disabled":
        artifact = wandb.Artifact(name="lora-adapter", type="model")
        artifact.add_dir(str(adapter_dir))
        wandb.log_artifact(artifact)

    wandb.finish()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", choices=["local", "remote"], default="local")
    args = p.parse_args()
    run(args.profile)


if __name__ == "__main__":
    main()
