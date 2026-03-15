from __future__ import annotations

import math
from typing import Optional

import torch
from datasets import load_dataset


def _get_model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def load_wikitext2_validation(
    tokenizer,
    sequence_length: int = 2048,
) -> torch.Tensor:
    """
    Load WikiText-2 raw validation text and tokenize it into one long tensor.

    Returns
    -------
    torch.Tensor
        Shape: [1, num_tokens]
    """
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join(dataset["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"]

    if input_ids.numel() < sequence_length + 1:
        raise ValueError(
            "Tokenized WikiText validation set is too short for the requested "
            f"sequence_length={sequence_length}."
        )

    return input_ids


@torch.no_grad()
def compute_perplexity(
    model: torch.nn.Module,
    tokenizer,
    sequence_length: int = 2048,
    stride: Optional[int] = None,
    max_eval_tokens: Optional[int] = None,
    show_progress: bool = True,
) -> float:
    """
    Compute perplexity on WikiText-2 validation.

    This uses the standard sliding-window causal LM evaluation:
    - feed windows of tokens to the model
    - only score the newly introduced tokens in each window

    Parameters
    ----------
    model : torch.nn.Module
        Hugging Face causal LM.
    tokenizer :
        Matching tokenizer.
    sequence_length : int
        Max context length per forward pass.
    stride : Optional[int]
        Number of new tokens introduced per step. Defaults to sequence_length.
        Set smaller than sequence_length for more standard overlapping eval.
    max_eval_tokens : Optional[int]
        Optionally truncate evaluation for faster debugging.
    show_progress : bool
        Whether to print progress.

    Returns
    -------
    float
        Perplexity.
    """
    model.eval()
    device = _get_model_device(model)

    if stride is None:
        stride = sequence_length

    if stride <= 0:
        raise ValueError("stride must be positive.")
    if sequence_length <= 1:
        raise ValueError("sequence_length must be greater than 1.")

    input_ids = load_wikitext2_validation(
        tokenizer=tokenizer,
        sequence_length=sequence_length,
    )

    if max_eval_tokens is not None:
        max_eval_tokens = min(max_eval_tokens, input_ids.shape[1])
        input_ids = input_ids[:, :max_eval_tokens]

    seq_len = input_ids.shape[1]
    nll_sum = 0.0
    n_tokens_scored = 0

    prev_end_loc = 0

    for begin_loc in range(0, seq_len - 1, stride):
        end_loc = min(begin_loc + sequence_length, seq_len)
        trg_len = end_loc - prev_end_loc
        if trg_len <= 0:
            continue

        input_chunk = input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_chunk.clone()

        # Only compute loss on the newly added tokens.
        # The model internally shifts labels by one token, so this marks the
        # prefix context as ignored while scoring the final trg_len tokens.
        target_ids[:, :-trg_len] = -100

        outputs = model(input_ids=input_chunk, labels=target_ids)
        neg_log_likelihood = outputs.loss.item() * trg_len

        nll_sum += neg_log_likelihood
        n_tokens_scored += trg_len

        prev_end_loc = end_loc

        if show_progress:
            print(
                f"[perplexity] processed tokens: {end_loc}/{seq_len} "
                f"(scored={n_tokens_scored})",
                end="\r",
                flush=True,
            )

        if end_loc == seq_len:
            break

    if show_progress:
        print()

    if n_tokens_scored == 0:
        raise RuntimeError("No tokens were scored during perplexity evaluation.")

    avg_nll = nll_sum / n_tokens_scored
    ppl = math.exp(avg_nll)
    return ppl