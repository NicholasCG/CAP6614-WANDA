from __future__ import annotations

import argparse
import json
import os
import time

import torch

from eval.perplexity import compute_perplexity
from models.load_model import load_model_and_tokenizer
from pruning.magnitude import magnitude_prune_model
from pruning.sparsegpt import SparseGPTPruner
from pruning.wanda import WandaPruner


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _count_nonzero_and_total(model: torch.nn.Module) -> tuple[int, int]:
    nonzero = 0
    total = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            w = module.weight.data
            nonzero += (w != 0).sum().item()
            total += w.numel()
    return nonzero, total


def _compute_linear_sparsity(model: torch.nn.Module) -> dict:
    nonzero, total = _count_nonzero_and_total(model)
    if total == 0:
        return {
            "linear_total_params": 0,
            "linear_nonzero_params": 0,
            "linear_zero_params": 0,
            "linear_sparsity": 0.0,
        }

    zero = total - nonzero
    return {
        "linear_total_params": total,
        "linear_nonzero_params": nonzero,
        "linear_zero_params": zero,
        "linear_sparsity": zero / total,
    }


def _save_json(obj: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _print_device_info(device) -> None:
    print(f"[main] resolved device = {device}")
    print(f"[main] torch.cuda.is_available() = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[main] gpu = {torch.cuda.get_device_name(0)}")


def run_magnitude(args) -> dict:
    run_start = time.time()

    print("[main] loading model/tokenizer...")
    load_start = time.time()
    model, tokenizer, device = load_model_and_tokenizer(
        model_name=args.model,
        dtype=args.dtype,
        device=args.device,
    )
    load_seconds = time.time() - load_start
    _print_device_info(device)
    print(f"[main] model/tokenizer loaded in {load_seconds:.2f}s")

    print("[main] computing dense perplexity...")
    dense_eval_start = time.time()
    dense_ppl = compute_perplexity(
        model=model,
        tokenizer=tokenizer,
        sequence_length=args.sequence_length,
        stride=args.perplexity_stride,
        max_eval_tokens=args.max_eval_tokens,
        show_progress=True,
    )
    dense_eval_seconds = time.time() - dense_eval_start
    print(f"[main] dense perplexity = {dense_ppl:.4f}")
    print(f"[main] dense perplexity eval took {dense_eval_seconds:.2f}s")

    print(f"[main] applying magnitude pruning at sparsity={args.sparsity} ...")
    prune_start = time.time()
    masks = magnitude_prune_model(model, sparsity=args.sparsity)
    prune_seconds = time.time() - prune_start
    print(f"[main] magnitude pruning done in {prune_seconds:.2f}s")

    sparsity_stats = _compute_linear_sparsity(model)

    print("[main] computing pruned perplexity...")
    pruned_eval_start = time.time()
    pruned_ppl = compute_perplexity(
        model=model,
        tokenizer=tokenizer,
        sequence_length=args.sequence_length,
        stride=args.perplexity_stride,
        max_eval_tokens=args.max_eval_tokens,
        show_progress=True,
    )
    pruned_eval_seconds = time.time() - pruned_eval_start
    print(f"[main] pruned perplexity = {pruned_ppl:.4f}")
    print(f"[main] pruned perplexity eval took {pruned_eval_seconds:.2f}s")

    total_seconds = time.time() - run_start

    return {
        "method": "magnitude",
        "model": args.model,
        "dtype": args.dtype,
        "device": str(device),
        "sequence_length": args.sequence_length,
        "perplexity_stride": args.perplexity_stride,
        "max_eval_tokens": args.max_eval_tokens,
        "target_sparsity": args.sparsity,
        "dense_perplexity": dense_ppl,
        "pruned_perplexity": pruned_ppl,
        "delta_perplexity": pruned_ppl - dense_ppl,
        "num_pruned_layers": len(masks),
        "model_load_seconds": load_seconds,
        "dense_eval_seconds": dense_eval_seconds,
        "prune_seconds": prune_seconds,
        "pruned_eval_seconds": pruned_eval_seconds,
        "total_seconds": total_seconds,
        **sparsity_stats,
    }


def run_wanda(args) -> dict:
    from data.calibration import get_c4_calibration_dataloader

    run_start = time.time()

    print("[main] loading model/tokenizer...")
    load_start = time.time()
    model, tokenizer, device = load_model_and_tokenizer(
        model_name=args.model,
        dtype=args.dtype,
        device=args.device,
    )
    load_seconds = time.time() - load_start
    _print_device_info(device)
    print(f"[main] model/tokenizer loaded in {load_seconds:.2f}s")

    print("[main] building C4 calibration dataloader...")
    calib_build_start = time.time()
    calib_loader = get_c4_calibration_dataloader(
        tokenizer=tokenizer,
        nsamples=args.nsamples,
        seed=args.seed,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
    )
    calib_build_seconds = time.time() - calib_build_start
    print(f"[main] calibration dataloader ready in {calib_build_seconds:.2f}s")

    print("[main] computing dense perplexity...")
    dense_eval_start = time.time()
    dense_ppl = compute_perplexity(
        model=model,
        tokenizer=tokenizer,
        sequence_length=args.sequence_length,
        stride=args.perplexity_stride,
        max_eval_tokens=args.max_eval_tokens,
        show_progress=True,
    )
    dense_eval_seconds = time.time() - dense_eval_start
    print(f"[main] dense perplexity = {dense_ppl:.4f}")
    print(f"[main] dense perplexity eval took {dense_eval_seconds:.2f}s")

    print(f"[main] collecting Wanda activation stats from C4 ({args.nsamples} samples)...")
    pruner = WandaPruner(model=model, sparsity=args.sparsity)

    collect_start = time.time()
    pruner.collect_activation_stats(dataloader=calib_loader, device=device)
    collect_seconds = time.time() - collect_start
    print(f"[main] Wanda activation stats collected in {collect_seconds:.2f}s")

    print(f"[main] applying Wanda pruning at sparsity={args.sparsity} ...")
    prune_start = time.time()
    masks = pruner.prune()
    prune_seconds = time.time() - prune_start
    print(f"[main] Wanda pruning done in {prune_seconds:.2f}s")

    sparsity_stats = _compute_linear_sparsity(model)

    print("[main] computing pruned perplexity...")
    pruned_eval_start = time.time()
    pruned_ppl = compute_perplexity(
        model=model,
        tokenizer=tokenizer,
        sequence_length=args.sequence_length,
        stride=args.perplexity_stride,
        max_eval_tokens=args.max_eval_tokens,
        show_progress=True,
    )
    pruned_eval_seconds = time.time() - pruned_eval_start
    print(f"[main] pruned perplexity = {pruned_ppl:.4f}")
    print(f"[main] pruned perplexity eval took {pruned_eval_seconds:.2f}s")

    total_seconds = time.time() - run_start

    return {
        "method": "wanda",
        "model": args.model,
        "dtype": args.dtype,
        "device": str(device),
        "sequence_length": args.sequence_length,
        "perplexity_stride": args.perplexity_stride,
        "max_eval_tokens": args.max_eval_tokens,
        "target_sparsity": args.sparsity,
        "nsamples": args.nsamples,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "dense_perplexity": dense_ppl,
        "pruned_perplexity": pruned_ppl,
        "delta_perplexity": pruned_ppl - dense_ppl,
        "num_pruned_layers": len(masks),
        "model_load_seconds": load_seconds,
        "calibration_build_seconds": calib_build_seconds,
        "dense_eval_seconds": dense_eval_seconds,
        "activation_collection_seconds": collect_seconds,
        "prune_seconds": prune_seconds,
        "pruned_eval_seconds": pruned_eval_seconds,
        "total_seconds": total_seconds,
        **sparsity_stats,
    }


def run_sparsegpt(args) -> dict:
    from data.calibration import get_c4_calibration_dataloader

    run_start = time.time()

    print("[main] loading model/tokenizer...")
    load_start = time.time()
    model, tokenizer, device = load_model_and_tokenizer(
        model_name=args.model,
        dtype=args.dtype,
        device=args.device,
    )
    load_seconds = time.time() - load_start
    _print_device_info(device)
    print(f"[main] model/tokenizer loaded in {load_seconds:.2f}s")

    print("[main] building C4 calibration dataloader...")
    calib_build_start = time.time()
    calib_loader = get_c4_calibration_dataloader(
        tokenizer=tokenizer,
        nsamples=args.nsamples,
        seed=args.seed,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
    )
    calib_build_seconds = time.time() - calib_build_start
    print(f"[main] calibration dataloader ready in {calib_build_seconds:.2f}s")

    print("[main] computing dense perplexity...")
    dense_eval_start = time.time()
    dense_ppl = compute_perplexity(
        model=model,
        tokenizer=tokenizer,
        sequence_length=args.sequence_length,
        stride=args.perplexity_stride,
        max_eval_tokens=args.max_eval_tokens,
        show_progress=True,
    )
    dense_eval_seconds = time.time() - dense_eval_start
    print(f"[main] dense perplexity = {dense_ppl:.4f}")
    print(f"[main] dense perplexity eval took {dense_eval_seconds:.2f}s")

    print(f"[main] collecting SparseGPT activation stats from C4 ({args.nsamples} samples)...")
    pruner = SparseGPTPruner(
        model=model,
        sparsity=args.sparsity,
        blocksize=args.blocksize,
        percdamp=args.percdamp,
        prune_n=args.prune_n,
        prune_m=args.prune_m,
    )

    collect_start = time.time()
    pruner.collect_activation_stats(dataloader=calib_loader, device=device)
    collect_seconds = time.time() - collect_start
    print(f"[main] SparseGPT activation stats collected in {collect_seconds:.2f}s")

    print(f"[main] applying SparseGPT pruning at sparsity={args.sparsity} ...")
    prune_start = time.time()
    masks = pruner.prune()
    prune_seconds = time.time() - prune_start
    print(f"[main] SparseGPT pruning done in {prune_seconds:.2f}s")

    sparsity_stats = _compute_linear_sparsity(model)

    print("[main] computing pruned perplexity...")
    pruned_eval_start = time.time()
    pruned_ppl = compute_perplexity(
        model=model,
        tokenizer=tokenizer,
        sequence_length=args.sequence_length,
        stride=args.perplexity_stride,
        max_eval_tokens=args.max_eval_tokens,
        show_progress=True,
    )
    pruned_eval_seconds = time.time() - pruned_eval_start
    print(f"[main] pruned perplexity = {pruned_ppl:.4f}")
    print(f"[main] pruned perplexity eval took {pruned_eval_seconds:.2f}s")

    total_seconds = time.time() - run_start

    return {
        "method": "sparsegpt",
        "model": args.model,
        "dtype": args.dtype,
        "device": str(device),
        "sequence_length": args.sequence_length,
        "perplexity_stride": args.perplexity_stride,
        "max_eval_tokens": args.max_eval_tokens,
        "target_sparsity": args.sparsity,
        "nsamples": args.nsamples,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "blocksize": args.blocksize,
        "percdamp": args.percdamp,
        "prune_n": args.prune_n,
        "prune_m": args.prune_m,
        "dense_perplexity": dense_ppl,
        "pruned_perplexity": pruned_ppl,
        "delta_perplexity": pruned_ppl - dense_ppl,
        "num_pruned_layers": len(masks),
        "model_load_seconds": load_seconds,
        "calibration_build_seconds": calib_build_seconds,
        "dense_eval_seconds": dense_eval_seconds,
        "activation_collection_seconds": collect_seconds,
        "prune_seconds": prune_seconds,
        "pruned_eval_seconds": pruned_eval_seconds,
        "total_seconds": total_seconds,
        **sparsity_stats,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Lean Wanda reproduction entrypoint.")

    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Hugging Face model name.",
    )
    parser.add_argument(
        "--prune_method",
        type=str,
        required=True,
        choices=["magnitude", "wanda", "sparsegpt"],
        help="Pruning method to run.",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.5,
        help="Target unstructured sparsity ratio.",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=2048,
        help="Sequence length for calibration/eval windows.",
    )
    parser.add_argument(
        "--perplexity_stride",
        type=int,
        default=None,
        help="Sliding-window stride for perplexity. Defaults to sequence_length.",
    )
    parser.add_argument(
        "--max_eval_tokens",
        type=int,
        default=None,
        help="Optional truncation for faster debugging.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
        help="Model dtype.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional explicit device, e.g. 'cuda' or 'cpu'.",
    )

    parser.add_argument(
        "--nsamples",
        type=int,
        default=128,
        help="Number of C4 calibration samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used in calibration sampling.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Calibration dataloader batch size.",
    )

    parser.add_argument(
        "--blocksize",
        type=int,
        default=128,
        help="SparseGPT block size.",
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="SparseGPT damping factor.",
    )
    parser.add_argument(
        "--prune_n",
        type=int,
        default=0,
        help="Optional n in n:m structured pruning. Use 0 for unstructured.",
    )
    parser.add_argument(
        "--prune_m",
        type=int,
        default=0,
        help="Optional m in n:m structured pruning. Use 0 for unstructured.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save run outputs.",
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save pruned HF model/tokenizer for the selected method.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    _ensure_dir(args.output_dir)

    if not (0.0 <= args.sparsity < 1.0):
        raise ValueError("--sparsity must satisfy 0.0 <= sparsity < 1.0")

    if args.prune_method == "magnitude":
        results = run_magnitude(args)

        if args.save_model:
            print("[main] saving pruned model/tokenizer...")
            save_start = time.time()
            model, tokenizer, _ = load_model_and_tokenizer(
                model_name=args.model,
                dtype=args.dtype,
                device=args.device,
            )
            magnitude_prune_model(model, sparsity=args.sparsity)
            save_dir = os.path.join(args.output_dir, "magnitude_model")
            _ensure_dir(save_dir)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            results["saved_model_dir"] = save_dir
            results["save_model_seconds"] = time.time() - save_start

    elif args.prune_method == "wanda":
        results = run_wanda(args)

        if args.save_model:
            print("[main] re-running Wanda once to save pruned model/tokenizer...")
            from data.calibration import get_c4_calibration_dataloader

            save_start = time.time()
            model, tokenizer, device = load_model_and_tokenizer(
                model_name=args.model,
                dtype=args.dtype,
                device=args.device,
            )
            calib_loader = get_c4_calibration_dataloader(
                tokenizer=tokenizer,
                nsamples=args.nsamples,
                seed=args.seed,
                sequence_length=args.sequence_length,
                batch_size=args.batch_size,
            )
            pruner = WandaPruner(model=model, sparsity=args.sparsity)
            pruner.collect_activation_stats(dataloader=calib_loader, device=device)
            pruner.prune()

            save_dir = os.path.join(args.output_dir, "wanda_model")
            _ensure_dir(save_dir)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            results["saved_model_dir"] = save_dir
            results["save_model_seconds"] = time.time() - save_start

    elif args.prune_method == "sparsegpt":
        results = run_sparsegpt(args)

        if args.save_model:
            print("[main] re-running SparseGPT once to save pruned model/tokenizer...")
            from data.calibration import get_c4_calibration_dataloader

            save_start = time.time()
            model, tokenizer, device = load_model_and_tokenizer(
                model_name=args.model,
                dtype=args.dtype,
                device=args.device,
            )
            calib_loader = get_c4_calibration_dataloader(
                tokenizer=tokenizer,
                nsamples=args.nsamples,
                seed=args.seed,
                sequence_length=args.sequence_length,
                batch_size=args.batch_size,
            )
            pruner = SparseGPTPruner(
                model=model,
                sparsity=args.sparsity,
                blocksize=args.blocksize,
                percdamp=args.percdamp,
                prune_n=args.prune_n,
                prune_m=args.prune_m,
            )
            pruner.collect_activation_stats(dataloader=calib_loader, device=device)
            pruner.prune()

            save_dir = os.path.join(args.output_dir, "sparsegpt_model")
            _ensure_dir(save_dir)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            results["saved_model_dir"] = save_dir
            results["save_model_seconds"] = time.time() - save_start

    else:
        raise ValueError(f"Unsupported prune method: {args.prune_method}")

    results_path = os.path.join(args.output_dir, f"{args.prune_method}_results.json")
    _save_json(results, results_path)

    print("\n[main] finished.")
    print(json.dumps(results, indent=2))
    print(f"[main] saved results to: {results_path}")


if __name__ == "__main__":
    main()