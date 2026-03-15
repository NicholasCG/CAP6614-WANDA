from __future__ import annotations

import os
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset


class CalibrationDataset(Dataset):
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [N, seq_len]")
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [N, seq_len]")
        if input_ids.shape != attention_mask.shape:
            raise ValueError("input_ids and attention_mask must have the same shape")

        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


def _default_cache_dir() -> Path:
    return Path("data") / "cache"


def _build_cache_path(
    nsamples: int,
    seed: int,
    sequence_length: int,
    tokenizer,
    cache_dir: str | os.PathLike | None = None,
) -> Path:
    root = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    root.mkdir(parents=True, exist_ok=True)

    tokenizer_name = getattr(tokenizer, "name_or_path", "tokenizer")
    safe_tokenizer_name = tokenizer_name.replace("/", "_").replace("\\", "_").replace(":", "_")

    filename = (
        f"c4_calibration_"
        f"tok-{safe_tokenizer_name}_"
        f"n-{nsamples}_"
        f"seed-{seed}_"
        f"seq-{sequence_length}.pt"
    )
    return root / filename


def sample_c4_sequences(
    tokenizer,
    nsamples: int = 128,
    seed: int = 0,
    sequence_length: int = 2048,
    cache_dir: str | os.PathLike | None = None,
    use_cache: bool = True,
) -> dict[str, torch.Tensor]:
    """
    Build fixed-length calibration sequences from the C4 train split.

    This function:
    - streams C4 so it doesn't download the full dataset
    - tokenizes text incrementally
    - packs tokens into exactly `sequence_length` chunks
    - caches the final tensor dataset to disk

    Returns
    -------
    dict with keys:
        input_ids: Tensor [nsamples, sequence_length]
        attention_mask: Tensor [nsamples, sequence_length]
    """
    cache_path = _build_cache_path(
        nsamples=nsamples,
        seed=seed,
        sequence_length=sequence_length,
        tokenizer=tokenizer,
        cache_dir=cache_dir,
    )

    if use_cache and cache_path.exists():
        print(f"[calibration] loading cached calibration samples from: {cache_path}")
        payload = torch.load(cache_path, map_location="cpu")
        return payload

    print("[calibration] cache miss; streaming C4 and building calibration samples...")
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)

    # Fixing runtime.
    skip_docs = int(seed) * 50
    if skip_docs > 0:
        print(f"[calibration] skipping {skip_docs} documents for seed={seed} ...")
        dataset = dataset.skip(skip_docs)

    packed_samples: list[torch.Tensor] = []
    token_buffer: list[int] = []

    docs_seen = 0
    tokens_buffered = 0

    for item in dataset:
        docs_seen += 1
        text = item.get("text", None)

        if text is None:
            continue
        text = text.strip()
        if not text:
            continue

        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]

        if not encoded:
            continue

        token_buffer.extend(encoded)
        tokens_buffered = len(token_buffer)

        while len(token_buffer) >= sequence_length and len(packed_samples) < nsamples:
            chunk = token_buffer[:sequence_length]
            del token_buffer[:sequence_length]
            packed_samples.append(torch.tensor(chunk, dtype=torch.long))

        if docs_seen % 25 == 0 or len(packed_samples) == nsamples:
            print(
                f"[calibration] docs_seen={docs_seen} "
                f"samples_built={len(packed_samples)}/{nsamples} "
                f"buffer_tokens={tokens_buffered}",
                end="\r",
                flush=True,
            )

        if len(packed_samples) >= nsamples:
            break

    print()

    if len(packed_samples) < nsamples:
        raise RuntimeError(
            f"Failed to collect enough calibration samples from C4. "
            f"Built {len(packed_samples)} but needed {nsamples}."
        )

    input_ids = torch.stack(packed_samples, dim=0)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    payload = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "meta": {
            "dataset": "allenai/c4",
            "config": "en",
            "split": "train",
            "nsamples": nsamples,
            "seed": seed,
            "sequence_length": sequence_length,
            "tokenizer": getattr(tokenizer, "name_or_path", "unknown"),
        },
    }

    if use_cache:
        torch.save(payload, cache_path)
        print(f"[calibration] saved calibration cache to: {cache_path}")

    return payload


def get_c4_calibration_dataloader(
    tokenizer,
    nsamples: int = 128,
    seed: int = 0,
    sequence_length: int = 2048,
    batch_size: int = 1,
    cache_dir: str | os.PathLike | None = None,
    use_cache: bool = True,
) -> DataLoader:
    payload = sample_c4_sequences(
        tokenizer=tokenizer,
        nsamples=nsamples,
        seed=seed,
        sequence_length=sequence_length,
        cache_dir=cache_dir,
        use_cache=use_cache,
    )

    dataset = CalibrationDataset(
        input_ids=payload["input_ids"],
        attention_mask=payload["attention_mask"],
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )