from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from typing import Dict, Iterable, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.calibration import get_c4_calibration_dataloader
from eval.perplexity import compute_perplexity


def _iter_prunable_linear_layers(model: torch.nn.Module) -> Iterable[Tuple[str, torch.nn.Linear]]:
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            yield name, module


def _find_block_index_from_name(name: str) -> int:
    """
    Best-effort parser for names like:
      model.layers.0.self_attn.q_proj
      model.layers.12.mlp.gate_proj

    Returns -1 if no block index is found.
    """
    parts = name.split(".")
    for i, part in enumerate(parts[:-1]):
        if part == "layers":
            try:
                return int(parts[i + 1])
            except (ValueError, IndexError):
                return -1
    return -1


def _categorize_submodule(name: str) -> str:
    if "self_attn" in name or "attention" in name or "attn" in name:
        return "attention"
    if "mlp" in name or "gate_proj" in name or "up_proj" in name or "down_proj" in name:
        return "mlp"
    return "other"


def _make_rowwise_mask_from_scores(scores: torch.Tensor, sparsity: float) -> torch.Tensor:
    """
    Given a [out_features, in_features] score matrix, prune the lowest-scoring
    fraction in each row independently.
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError("sparsity must satisfy 0.0 <= sparsity < 1.0")

    out_features, in_features = scores.shape
    k = int(in_features * sparsity)

    if k == 0:
        return torch.ones_like(scores, dtype=scores.dtype, device=scores.device)

    mask = torch.ones_like(scores, dtype=scores.dtype, device=scores.device)

    for i in range(out_features):
        row = scores[i]
        prune_idx = torch.topk(row, k=k, largest=False).indices
        mask[i, prune_idx] = 0

    return mask


@torch.no_grad()
def _collect_input_norms_for_named_layers(
    model: torch.nn.Module,
    dataloader,
    target_layer_names: List[str],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Collect L2 norms of input activation channels for selected linear layers.
    """
    target_set = set(target_layer_names)
    input_norms: Dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(layer_name: str):
        def hook(module, inp, output):
            x = inp[0]
            x = x.reshape(-1, x.shape[-1]).detach()
            norm = torch.norm(x, p=2, dim=0)

            if layer_name not in input_norms:
                input_norms[layer_name] = norm
            else:
                input_norms[layer_name] += norm

        return hook

    for name, module in _iter_prunable_linear_layers(model):
        if name in target_set:
            handles.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            model(input_ids=input_ids)

    for handle in handles:
        handle.remove()

    missing = [name for name in target_layer_names if name not in input_norms]
    if missing:
        raise RuntimeError(f"Failed to collect activation norms for layers: {missing}")

    return input_norms


def _apply_wanda_to_selected_layers(
    model: torch.nn.Module,
    layer_names: List[str],
    input_norms: Dict[str, torch.Tensor],
    sparsity: float,
) -> Dict[str, torch.Tensor]:
    """
    Prune only the selected layers in-place using the Wanda score:
        score = |W| * ||x||
    """
    masks: Dict[str, torch.Tensor] = {}

    for name, module in _iter_prunable_linear_layers(model):
        if name not in layer_names:
            continue

        W = module.weight.data
        act_norm = input_norms[name].to(W.device, dtype=W.dtype)
        scores = W.abs() * act_norm.unsqueeze(0)
        mask = _make_rowwise_mask_from_scores(scores=scores, sparsity=sparsity)

        module.weight.data.mul_(mask)
        masks[name] = mask

    return masks


def _group_layers(
    model: torch.nn.Module,
    mode: str,
) -> Dict[str, List[str]]:
    """
    Build groups of layer names for sensitivity analysis.

    mode='block'
        One group per transformer block, containing all linear layers in that block.

    mode='block_submodule'
        Separate attention and mlp groups inside each block.

    mode='layer'
        Each individual linear layer gets its own group.
    """
    groups: Dict[str, List[str]] = {}

    for name, module in _iter_prunable_linear_layers(model):
        block_idx = _find_block_index_from_name(name)
        submodule = _categorize_submodule(name)

        if mode == "layer":
            group_name = name
        elif mode == "block":
            group_name = f"block_{block_idx:02d}" if block_idx >= 0 else "block_unknown"
        elif mode == "block_submodule":
            if block_idx >= 0:
                group_name = f"block_{block_idx:02d}_{submodule}"
            else:
                group_name = f"block_unknown_{submodule}"
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        groups.setdefault(group_name, []).append(name)

    return groups


def _load_model_and_tokenizer(model_name: str, dtype: str):
    torch_dtype = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[dtype.lower()]

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )

    return model, tokenizer


def run_layer_sensitivity(
    model_name: str,
    output_dir: str,
    sparsity: float = 0.5,
    sequence_length: int = 2048,
    nsamples: int = 128,
    seed: int = 0,
    dtype: str = "float16",
    group_mode: str = "block",
    perplexity_stride: int | None = None,
    max_eval_tokens: int | None = None,
) -> List[dict]:
    os.makedirs(output_dir, exist_ok=True)

    print("[1/5] Loading model and tokenizer...")
    base_model, tokenizer = _load_model_and_tokenizer(model_name=model_name, dtype=dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model.to(device)
    base_model.eval()

    print("[2/5] Building calibration dataloader...")
    calib_loader = get_c4_calibration_dataloader(
        tokenizer=tokenizer,
        nsamples=nsamples,
        seed=seed,
        sequence_length=sequence_length,
        batch_size=1,
    )

    print("[3/5] Computing dense baseline perplexity...")
    dense_ppl = compute_perplexity(
        model=base_model,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        stride=perplexity_stride,
        max_eval_tokens=max_eval_tokens,
        show_progress=True,
    )
    print(f"[baseline] dense perplexity = {dense_ppl:.4f}")

    print("[4/5] Creating sensitivity groups...")
    groups = _group_layers(base_model, mode=group_mode)
    print(f"[groups] total groups = {len(groups)}")

    results: List[dict] = []

    print("[5/5] Running per-group Wanda pruning experiments...")
    for idx, (group_name, layer_names) in enumerate(groups.items(), start=1):
        print(f"\n[group {idx}/{len(groups)}] {group_name}")
        print(f"  collecting activation stats for {len(layer_names)} layer(s)...")

        # Fresh model copy per experiment to avoid cumulative damage.
        model_copy = copy.deepcopy(base_model)
        model_copy.to(device)
        model_copy.eval()

        input_norms = _collect_input_norms_for_named_layers(
            model=model_copy,
            dataloader=calib_loader,
            target_layer_names=layer_names,
            device=device,
        )

        _apply_wanda_to_selected_layers(
            model=model_copy,
            layer_names=layer_names,
            input_norms=input_norms,
            sparsity=sparsity,
        )

        ppl = compute_perplexity(
            model=model_copy,
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            stride=perplexity_stride,
            max_eval_tokens=max_eval_tokens,
            show_progress=False,
        )

        delta = ppl - dense_ppl
        result = {
            "group_name": group_name,
            "group_mode": group_mode,
            "num_layers_pruned": len(layer_names),
            "layer_names": layer_names,
            "sparsity": sparsity,
            "dense_perplexity": dense_ppl,
            "pruned_perplexity": ppl,
            "delta_perplexity": delta,
        }
        results.append(result)

        print(f"  pruned perplexity = {ppl:.4f}")
        print(f"  delta perplexity  = {delta:.4f}")

        del model_copy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    json_path = os.path.join(output_dir, f"layer_sensitivity_{group_mode}.json")
    csv_path = os.path.join(output_dir, f"layer_sensitivity_{group_mode}.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group_name",
                "group_mode",
                "num_layers_pruned",
                "sparsity",
                "dense_perplexity",
                "pruned_perplexity",
                "delta_perplexity",
                "layer_names",
            ],
        )
        writer.writeheader()
        for row in results:
            row = dict(row)
            row["layer_names"] = ";".join(row["layer_names"])
            writer.writerow(row)

    print(f"\nSaved JSON results to: {json_path}")
    print(f"Saved CSV results to:  {csv_path}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run Wanda layer sensitivity analysis.")
    parser.add_argument("--model", type=str, required=True, help="HF model name, e.g. meta-llama/Llama-2-7b-hf")
    parser.add_argument("--output_dir", type=str, default="results/layer_sensitivity")
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--sequence_length", type=int, default=2048)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
    )
    parser.add_argument(
        "--group_mode",
        type=str,
        default="block",
        choices=["layer", "block", "block_submodule"],
        help=(
            "layer: prune one linear layer at a time; "
            "block: prune all linear layers in one transformer block at a time; "
            "block_submodule: prune attention or mlp portions within a block."
        ),
    )
    parser.add_argument(
        "--perplexity_stride",
        type=int,
        default=None,
        help="Sliding-window stride for perplexity eval. Defaults to sequence_length.",
    )
    parser.add_argument(
        "--max_eval_tokens",
        type=int,
        default=None,
        help="Optional truncation for faster debugging.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_layer_sensitivity(
        model_name=args.model,
        output_dir=args.output_dir,
        sparsity=args.sparsity,
        sequence_length=args.sequence_length,
        nsamples=args.nsamples,
        seed=args.seed,
        dtype=args.dtype,
        group_mode=args.group_mode,
        perplexity_stride=args.perplexity_stride,
        max_eval_tokens=args.max_eval_tokens,
    )