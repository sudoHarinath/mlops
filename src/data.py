from datasets import Dataset, load_dataset

from .config import DatasetConfig


def format_example(example: dict) -> dict:
    instruction = example["instruction"]
    intent = example["intent"]
    category = example["category"]
    response = example["response"]

    summary = response.strip().split("\n")[0][:80]
    target = (
        '{"intent": "' + intent + '", '
        '"category": "' + category + '", '
        '"summary": "' + summary.replace('"', "'") + '"}'
    )

    text = (
        "<|user|>\n"
        f"Classify this support ticket and return JSON: {instruction}\n"
        "<|assistant|>\n"
        f"{target}"
    )
    return {"text": text}


def prepare(cfg: DatasetConfig) -> Dataset:
    ds = load_dataset(cfg.name, split="train")
    n = min(cfg.sample_size, len(ds))
    ds = ds.shuffle(seed=cfg.seed).select(range(n))
    return ds.map(format_example, remove_columns=ds.column_names)
