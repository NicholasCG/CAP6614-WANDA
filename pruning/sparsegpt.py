from __future__ import annotations

import math
import time
from typing import Dict

import torch
import torch.nn as nn


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def find_linear_layers(module: nn.Module, prefix: str = "") -> Dict[str, nn.Linear]:
    """
    Recursively collect all nn.Linear modules under `module`.
    """
    result: Dict[str, nn.Linear] = {}

    if isinstance(module, nn.Linear):
        result[prefix] = module
        return result

    for name, child in module.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        result.update(find_linear_layers(child, child_prefix))

    return result


def _extract_batch(batch):
    """
    Support dict batches from your calibration loader.
    """
    if isinstance(batch, dict):
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
        return input_ids, attention_mask

    if isinstance(batch, (tuple, list)):
        input_ids = batch[0]
        attention_mask = batch[1] if len(batch) > 1 else None
        return input_ids, attention_mask

    raise TypeError(f"Unsupported batch type: {type(batch)}")


def _maybe_get_hf_device_map(model):
    return getattr(model, "hf_device_map", None)


def _build_decoder_kwargs(
    model,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
) -> dict:
    """
    Build kwargs for direct decoder-layer calls across HF Llama variants.
    """
    kwargs = {}

    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask

    if position_ids is None:
        position_ids = torch.arange(
            hidden_states.shape[1],
            device=hidden_states.device,
            dtype=torch.long,
        ).unsqueeze(0)

    kwargs["position_ids"] = position_ids

    rotary_emb = getattr(model.model, "rotary_emb", None)
    if rotary_emb is not None:
        try:
            position_embeddings = rotary_emb(hidden_states, position_ids)
        except TypeError:
            position_embeddings = rotary_emb(hidden_states, position_ids=position_ids)
        kwargs["position_embeddings"] = position_embeddings

    return kwargs


class SparseGPT:
    """
    Per-linear-layer SparseGPT helper.

    This stays close to the core algorithm:
    - accumulate approximate Hessian / input covariance
    - damp and factorize
    - prune with blockwise second-order error compensation
    """

    def __init__(self, layer: nn.Linear):
        if not isinstance(layer, nn.Linear):
            raise TypeError("SparseGPT currently supports nn.Linear only.")

        self.layer = layer
        self.dev = layer.weight.device

        W = layer.weight.data
        self.rows = W.shape[0]
        self.columns = W.shape[1]

        self.H = torch.zeros(
            (self.columns, self.columns),
            device=self.dev,
            dtype=torch.float32,
        )
        self.nsamples = 0

    def add_batch(self, inp, out=None):
        """
        Accumulate input covariance statistics.
        """
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)

        if inp.dim() == 3:
            inp = inp.reshape((-1, inp.shape[-1]))

        inp = inp.t().float()
        tmp = inp.shape[1]

        if tmp == 0:
            return

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        inp = math.sqrt(2.0 / self.nsamples) * inp
        self.H += inp.matmul(inp.t())

    @torch.no_grad()
    def fasterprune(
        self,
        sparsity,
        prune_n=0,
        prune_m=0,
        blocksize=128,
        percdamp=0.01,
    ):
        W = self.layer.weight.data.clone().float()
        H = self.H
        self.H = None

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp

        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prune_n == 0:
                scores = W1.pow(2) / (torch.diag(Hinv1).reshape((1, -1)).pow(2))
                thresh = torch.sort(scores.flatten())[0][int(scores.numel() * sparsity)]
                mask1 = scores <= thresh
            else:
                mask1 = torch.zeros_like(W1, dtype=torch.bool)

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prune_n != 0 and i % prune_m == 0:
                    local_end = min(i + prune_m, count)
                    scores = (
                        W1[:, i:local_end].pow(2)
                        / torch.diag(Hinv1)[i:local_end].reshape((1, -1)).pow(2)
                    )
                    idx = torch.topk(scores, prune_n, dim=1, largest=False).indices
                    mask1.scatter_(1, i + idx, True)

                q = w.clone()
                q[mask1[:, i]] = 0

                Q1[:, i] = q

                err1 = (w - q) / d
                if i + 1 < count:
                    W1[:, i + 1:] -= err1.unsqueeze(1).matmul(
                        Hinv1[i, i + 1:].unsqueeze(0)
                    )
                Err1[:, i] = err1

            W[:, i1:i2] = Q1

            if i2 < self.columns:
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        self.layer.weight.data.copy_(W.to(self.layer.weight.data.dtype))

    def free(self):
        self.H = None
        torch.cuda.empty_cache()


class SparseGPTPruner:
    """
    Project-level SparseGPT wrapper matching your other pruning modules.

    Usage:
        pruner = SparseGPTPruner(model, sparsity=0.5)
        pruner.collect_activation_stats(dataloader, device)
        masks = pruner.prune()

    Unlike your previous version, this does NOT build SparseGPT Hessians for
    the whole model at once. It stores only the calibration activations needed
    to replay the transformer block sequence, then prunes block-by-block.
    """

    def __init__(
        self,
        model,
        sparsity,
        blocksize=128,
        percdamp=0.01,
        prune_n=0,
        prune_m=0,
    ):
        self.model = model
        self.sparsity = sparsity
        self.blocksize = blocksize
        self.percdamp = percdamp
        self.prune_n = prune_n
        self.prune_m = prune_m

        self.device = None
        self.inps = None
        self.outs = None
        self.attention_mask = None
        self.position_ids = None
        self.nsamples = 0

    @torch.no_grad()
    def collect_activation_stats(self, dataloader, device):
        """
        Capture the inputs to the first transformer block for the calibration set.
        This is the memory-safe SparseGPT setup.
        """
        self.model.eval()
        self.device = device

        layers = self.model.model.layers
        hf_device_map = _maybe_get_hf_device_map(self.model)

        first_dev = device
        if hf_device_map is not None and "model.embed_tokens" in hf_device_map:
            first_dev = hf_device_map["model.embed_tokens"]

        try:
            self.nsamples = len(dataloader.dataset)
        except Exception:
            self.nsamples = len(dataloader)

        dtype = next(self.model.parameters()).dtype
        hidden_size = self.model.config.hidden_size
        seqlen = self.model.seqlen

        self.inps = torch.zeros(
            (self.nsamples, seqlen, hidden_size),
            dtype=dtype,
            device=first_dev,
        )
        self.outs = torch.zeros_like(self.inps)

        cache = {
            "i": 0,
            "attention_mask": None,
            "position_ids": None,
        }

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                if cache["i"] >= self_inps.shape[0]:
                    raise ValueError

                self_inps[cache["i"]] = inp
                cache["i"] += 1
                cache["attention_mask"] = kwargs.get("attention_mask", None)
                cache["position_ids"] = kwargs.get("position_ids", None)
                raise ValueError

        self_inps = self.inps
        original_use_cache = self.model.config.use_cache
        self.model.config.use_cache = False

        layers[0] = Catcher(layers[0])

        for i, batch in enumerate(dataloader, start=1):
            input_ids, attention_mask = _extract_batch(batch)

            batch_size = input_ids.shape[0]
            for b in range(batch_size):
                if cache["i"] >= self.nsamples:
                    break

                kwargs = {}
                if attention_mask is not None:
                    kwargs["attention_mask"] = attention_mask[b:b + 1].to(first_dev)

                try:
                    self.model(input_ids[b:b + 1].to(first_dev), **kwargs)
                except ValueError:
                    pass

            print(
                f"[sparsegpt] collected calibration batch {i}",
                end="\r",
                flush=True,
            )

            if cache["i"] >= self.nsamples:
                break

        print()

        layers[0] = layers[0].module
        self.model.config.use_cache = original_use_cache

        self.attention_mask = cache["attention_mask"]
        self.position_ids = cache["position_ids"]

        if self.position_ids is None:
            self.position_ids = torch.arange(
                seqlen,
                device=self.inps.device,
                dtype=torch.long,
            ).unsqueeze(0)

        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.to(self.inps.device)

    @torch.no_grad()
    def prune(self):
        """
        Apply SparseGPT block-by-block and return a lightweight masks dict.
        """
        if self.inps is None:
            raise RuntimeError(
                "SparseGPTPruner.collect_activation_stats(...) must be called before prune()."
            )

        layers = self.model.model.layers
        hf_device_map = _maybe_get_hf_device_map(self.model)
        original_use_cache = self.model.config.use_cache
        self.model.config.use_cache = False

        masks = {}

        for layer_idx, layer in enumerate(layers):
            layer_dev = self.device
            if hf_device_map is not None and f"model.layers.{layer_idx}" in hf_device_map:
                layer_dev = hf_device_map[f"model.layers.{layer_idx}"]

                self.inps = self.inps.to(layer_dev)
                self.outs = self.outs.to(layer_dev)
                if self.attention_mask is not None:
                    self.attention_mask = self.attention_mask.to(layer_dev)
                if self.position_ids is not None:
                    self.position_ids = self.position_ids.to(layer_dev)

            subset = find_linear_layers(layer)
            gpts = {name: SparseGPT(mod) for name, mod in subset.items()}

            def add_batch(name):
                def hook(module, inp, output):
                    gpts[name].add_batch(inp[0].data, output.data if output is not None else None)
                return hook

            handles = []
            for name in gpts:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            # Pass current block once to collect second-order stats.
            for j in range(self.nsamples):
                hidden_states = self.inps[j].unsqueeze(0)
                kwargs = _build_decoder_kwargs(
                    model=self.model,
                    hidden_states=hidden_states,
                    attention_mask=self.attention_mask,
                    position_ids=self.position_ids,
                )
                self.outs[j] = layer(hidden_states, **kwargs)[0]

            for h in handles:
                h.remove()

            # Prune the current block's linear sublayers.
            for name, gpt in gpts.items():
                full_name = f"model.layers.{layer_idx}.{name}" if name else f"model.layers.{layer_idx}"
                print(f"[sparsegpt] pruning layer {full_name}")

                start = time.time()
                gpt.fasterprune(
                    self.sparsity,
                    prune_n=self.prune_n,
                    prune_m=self.prune_m,
                    blocksize=self.blocksize,
                    percdamp=self.percdamp,
                )
                elapsed = time.time() - start

                masks[full_name] = True
                gpt.free()

                print(f"[sparsegpt] finished {full_name} in {elapsed:.2f}s")

            # Replay the now-pruned block to produce inputs for the next block.
            for j in range(self.nsamples):
                hidden_states = self.inps[j].unsqueeze(0)
                kwargs = _build_decoder_kwargs(
                    model=self.model,
                    hidden_states=hidden_states,
                    attention_mask=self.attention_mask,
                    position_ids=self.position_ids,
                )
                self.outs[j] = layer(hidden_states, **kwargs)[0]

            self.inps, self.outs = self.outs, self.inps

            del gpts
            torch.cuda.empty_cache()

        self.model.config.use_cache = original_use_cache

        # Free calibration tensors now that pruning is done.
        self.inps = None
        self.outs = None
        self.attention_mask = None
        self.position_ids = None
        torch.cuda.empty_cache()

        return masks