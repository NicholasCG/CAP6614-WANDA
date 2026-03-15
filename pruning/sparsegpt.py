import math
import time
from collections import defaultdict

import torch
import torch.nn as nn


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class SparseGPT:
    """
    Per-layer SparseGPT helper.
    
    Accumulate an input covariance/Hessian approximation, then do
    blockwise second-order pruning with error compensation.
    """

    def __init__(self, layer):
        if not isinstance(layer, nn.Linear):
            raise TypeError("SparseGPT in this project currently supports nn.Linear only.")

        self.layer = layer
        self.dev = self.layer.weight.device

        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]

        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        """
        Accumulate approximate Hessian statistics from layer inputs.

        For Linear layers:
          inp expected shape after hook is typically [batch, seq, hidden]
          which we flatten to [tokens, hidden], then transpose to [hidden, tokens]
        """
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)

        tmp = inp.shape[0]

        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        inp = math.sqrt(2 / self.nsamples) * inp.float()
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
        """
        SparseGPT blockwise second-order pruning.
        """
        W = self.layer.weight.data.clone().float()

        H = self.H
        del self.H

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp

        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prune_n == 0:
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1, dtype=torch.bool)

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prune_n != 0 and i % prune_m == 0:
                    tmp = (
                        W1[:, i:(i + prune_m)] ** 2
                        / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                    )
                    mask1.scatter_(
                        1,
                        i + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                        True,
                    )

                q = w.clone()
                q[mask1[:, i]] = 0

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            losses += torch.sum(Losses1, 1) / 2

            if i2 < self.columns:
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

    def free(self):
        self.H = None
        torch.cuda.empty_cache()


class SparseGPTPruner:
    """
    Project-level SparseGPT wrapper.

    Usage:
        pruner = SparseGPTPruner(model, sparsity=0.5)
        pruner.collect_activation_stats(dataloader, device)
        masks = pruner.prune()
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

        self.gpts = {}
        self.handles = []

    def _hook_fn(self, name):
        def hook(module, inp, output):
            self.gpts[name].add_batch(inp[0].data, output.data if output is not None else None)
        return hook

    def register_hooks(self):
        self.gpts = {}
        self.handles = []

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                self.gpts[name] = SparseGPT(module)
                handle = module.register_forward_hook(self._hook_fn(name))
                self.handles.append(handle)

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    @torch.no_grad()
    def collect_activation_stats(self, dataloader, device):
        """
        Run calibration data through the model while accumulating per-layer
        SparseGPT statistics.
        """
        self.register_hooks()
        self.model.eval()

        for i, batch in enumerate(dataloader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", None)

            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                self.model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                self.model(input_ids=input_ids)

            print(
                f"[sparsegpt] collected calibration batch {i}",
                end="\r",
                flush=True,
            )

        print()
        self.remove_hooks()

    @torch.no_grad()
    def _build_mask(self, layer):
        """
        Build a boolean mask of zeroed weights after pruning by comparing the
        current weights to zero.
        """
        return layer.weight.data == 0

    @torch.no_grad()
    def prune(self):
        """
        Apply SparseGPT to every Linear layer in-place and return masks.
        """
        masks = {}

        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            print(f"[sparsegpt] pruning layer {name}")

            start = time.time()
            self.gpts[name].fasterprune(
                self.sparsity,
                prune_n=self.prune_n,
                prune_m=self.prune_m,
                blocksize=self.blocksize,
                percdamp=self.percdamp,
            )
            elapsed = time.time() - start

            masks[name] = self._build_mask(module)
            self.gpts[name].free()

            print(f"[sparsegpt] finished {name} in {elapsed:.2f}s")

        return masks