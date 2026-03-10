from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


Tensor = torch.Tensor


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def _move_to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_to_device(v, device) for v in obj)
    return obj


@torch.no_grad()
def _run_model_forward(model: nn.Module, batch: Any) -> Any:
    """Best-effort forward wrapper for tensor / tuple / dict batches."""
    if isinstance(batch, dict):
        return model(**batch)
    if isinstance(batch, (list, tuple)):
        return model(*batch)
    return model(batch)


@torch.no_grad()
def _flatten_linear_input(inp: Tensor, in_features: int) -> Tensor:
    """Convert arbitrary linear input shape (..., in_features) -> [tokens, in_features]."""
    if inp.dim() == 1:
        inp = inp.unsqueeze(0)
    if inp.shape[-1] != in_features:
        raise ValueError(
            f"Expected last dim {in_features} for linear input, got {tuple(inp.shape)}"
        )
    return inp.reshape(-1, in_features)


@torch.no_grad()
def _apply_nm_mask_from_scores(scores: Tensor, prunen: int, prunem: int) -> Tensor:
    """Return boolean mask where True means 'prune this weight'.

    scores: [out_features, in_features], lower score = less important.
    In every contiguous group of M columns, prune N smallest per row.
    """
    if prunen <= 0 or prunem <= 0:
        raise ValueError("Both prunen and prunem must be > 0 for N:M sparsity.")
    if prunen > prunem:
        raise ValueError("prunen must be <= prunem.")

    rows, cols = scores.shape
    mask = torch.zeros_like(scores, dtype=torch.bool)
    for start in range(0, cols, prunem):
        end = min(start + prunem, cols)
        group = scores[:, start:end]
        k = min(prunen, group.shape[1])
        idx = torch.topk(group, k=k, dim=1, largest=False).indices
        mask.scatter_(1, idx + start, True)
    return mask


@torch.no_grad()
def _count_zeros(weight: Tensor) -> Tuple[int, int]:
    zeros = int((weight == 0).sum().item())
    total = weight.numel()
    return zeros, total


# --------------------------------------------------------------------------------------
# Calibration statistics
# --------------------------------------------------------------------------------------

class LayerStats:
    def __init__(self, input_l2_norm: Optional[Tensor] = None, hessian: Optional[Tensor] = None, num_tokens: int = 0):
        self.input_l2_norm = input_l2_norm
        self.hessian = hessian
        self.num_tokens = num_tokens


class LinearCalibrationCollector:
    """Collects Wanda feature norms and SparseGPT Hessian statistics for one nn.Linear."""

    def __init__(self, layer: nn.Linear, collect_wanda: bool = True, collect_hessian: bool = True):
        self.layer = layer
        self.collect_wanda = collect_wanda
        self.collect_hessian = collect_hessian
        self._sum_sq: Optional[Tensor] = None
        self._xtx: Optional[Tensor] = None
        self._num_tokens: int = 0

    @torch.no_grad()
    def add_batch(self, inp: Tensor) -> None:
        x = _flatten_linear_input(inp, self.layer.in_features).to(dtype=torch.float32)
        n = x.shape[0]
        self._num_tokens += n

        if self.collect_wanda:
            if self._sum_sq is None:
                self._sum_sq = torch.zeros(
                    self.layer.in_features, device=x.device, dtype=torch.float32
                )
            self._sum_sq += (x * x).sum(dim=0)

        if self.collect_hessian:
            if self._xtx is None:
                self._xtx = torch.zeros(
                    self.layer.in_features,
                    self.layer.in_features,
                    device=x.device,
                    dtype=torch.float32,
                )
            self._xtx += x.T @ x

    @torch.no_grad()
    def as_stats(self) -> LayerStats:
        norm = None
        if self._sum_sq is not None:
            norm = torch.sqrt(torch.clamp(self._sum_sq, min=0))
        hessian = None
        if self._xtx is not None:
            denom = max(self._num_tokens, 1)
            hessian = self._xtx / float(denom)
        return LayerStats(input_l2_norm=norm, hessian=hessian, num_tokens=self._num_tokens)


# --------------------------------------------------------------------------------------
# Pruners
# --------------------------------------------------------------------------------------

class BasePruner:
    def __init__(self, layer: nn.Linear):
        self.layer = layer

    @torch.no_grad()
    def prune(
        self,
        sparsity: float,
        *,
        prunen: int = 0,
        prunem: int = 0,
        **kwargs: Any,
    ) -> Tensor:
        raise NotImplementedError


class MagnitudePruner(BasePruner):
    """Magnitude pruning baseline used in the Wanda paper.

    Unstructured case: prune globally within the layer by |W|.
    Structured N:M case: in each contiguous M-group per output row, prune N smallest |W|.
    """

    @torch.no_grad()
    def prune(
        self,
        sparsity: float,
        *,
        prunen: int = 0,
        prunem: int = 0,
        **kwargs: Any,
    ) -> Tensor:
        W = self.layer.weight.data
        scores = W.abs().float()

        if prunen > 0 and prunem > 0:
            mask = _apply_nm_mask_from_scores(scores, prunen, prunem)
        else:
            k = int(scores.numel() * sparsity)
            if k <= 0:
                return W
            flat = scores.flatten()
            prune_idx = torch.topk(flat, k=k, largest=False).indices
            mask = torch.zeros_like(flat, dtype=torch.bool)
            mask[prune_idx] = True
            mask = mask.view_as(W)

        W[mask] = 0
        return W


class WandaPruner(BasePruner):
    """Wanda = |W_ij| * ||X_j||_2, compared per output row for unstructured pruning.

    The paper defines the score as:
        S_ij = |W_ij| * ||X_j||_2
    where ||X_j||_2 is the L2 norm of the j-th input feature over calibration tokens.
    """

    def __init__(self, layer: nn.Linear, stats: LayerStats):
        super().__init__(layer)
        if stats.input_l2_norm is None:
            raise ValueError("WandaPruner requires input L2 norms from calibration data.")
        self.stats = stats

    @torch.no_grad()
    def prune(
        self,
        sparsity: float,
        *,
        prunen: int = 0,
        prunem: int = 0,
        **kwargs: Any,
    ) -> Tensor:
        W = self.layer.weight.data
        feature_norm = self.stats.input_l2_norm.to(W.device, dtype=torch.float32)
        scores = W.abs().float() * feature_norm.unsqueeze(0)

        if prunen > 0 and prunem > 0:
            mask = _apply_nm_mask_from_scores(scores, prunen, prunem)
        else:
            per_row = int(W.shape[1] * sparsity)
            if per_row <= 0:
                return W
            idx = torch.topk(scores, k=per_row, dim=1, largest=False).indices
            mask = torch.zeros_like(W, dtype=torch.bool)
            mask.scatter_(1, idx, True)

        W[mask] = 0
        return W


class SparseGPTPruner(BasePruner):
    """Blockwise SparseGPT implementation for nn.Linear.

    This follows the standard one-shot local reconstruction style used by SparseGPT:
    build a Hessian / covariance estimate from calibration activations, choose a sparse mask,
    and compensate the remaining weights with blockwise second-order updates.

    Notes:
    - Supports unstructured pruning and N:M semi-structured pruning.
    - This implementation targets plain nn.Linear layers.
    - It is intentionally self-contained and omits quantization hooks from the reference repo.
    """

    def __init__(self, layer: nn.Linear, stats: LayerStats):
        super().__init__(layer)
        if stats.hessian is None:
            raise ValueError("SparseGPTPruner requires Hessian statistics from calibration data.")
        self.stats = stats

    @torch.no_grad()
    def prune(
        self,
        sparsity: float,
        *,
        prunen: int = 0,
        prunem: int = 0,
        blocksize: int = 128,
        percdamp: float = 0.01,
        mask: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        W = self.layer.weight.data.clone().float()
        device = W.device
        columns = W.shape[1]

        H = self.stats.hessian.to(device=device, dtype=torch.float32).clone()
        diag = torch.diag(H)
        dead = diag == 0
        if dead.any():
            H[dead, dead] = 1.0
            W[:, dead] = 0

        damp = percdamp * torch.mean(torch.diag(H))
        diag_idx = torch.arange(columns, device=device)
        H[diag_idx, diag_idx] += damp

        # H^{-1/2} factor used in the reference implementation.
        # We obtain it as an upper-triangular factor U such that H^{-1} = U^T U.
        try:
            L = torch.linalg.cholesky(H)
            Hinv = torch.cholesky_inverse(L)
            U = torch.linalg.cholesky(Hinv, upper=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "SparseGPT Hessian factorization failed. Try increasing percdamp or using more calibration data."
            ) from exc

        losses = torch.zeros_like(W)

        for c_start in range(0, columns, blocksize):
            c_end = min(c_start + blocksize, columns)
            count = c_end - c_start

            W1 = W[:, c_start:c_end].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            U1 = U[c_start:c_end, c_start:c_end]

            if prunen == 0:
                if mask is not None:
                    mask1 = mask[:, c_start:c_end].clone()
                else:
                    base_scores = W1.pow(2) / (torch.diag(U1).reshape(1, -1).pow(2) + 1e-12)
                    k = int(base_scores.numel() * sparsity)
                    if k <= 0:
                        mask1 = torch.zeros_like(W1, dtype=torch.bool)
                    else:
                        thresh = torch.topk(base_scores.flatten(), k=k, largest=False).values.max()
                        mask1 = base_scores <= thresh
            else:
                mask1 = torch.zeros_like(W1, dtype=torch.bool)

            for i in range(count):
                w = W1[:, i]
                d = U1[i, i]

                if prunen > 0 and prunem > 0 and (i % prunem == 0):
                    grp_end = min(i + prunem, count)
                    group_scores = W1[:, i:grp_end].pow(2) / (
                        torch.diag(U1)[i:grp_end].reshape(1, -1).pow(2) + 1e-12
                    )
                    k = min(prunen, grp_end - i)
                    idx = torch.topk(group_scores, k=k, dim=1, largest=False).indices
                    mask1.scatter_(1, idx + i, True)

                q = w.clone()
                q[mask1[:, i]] = 0

                Losses1[:, i] = (w - q).pow(2) / (d * d + 1e-12)
                err = (w - q) / (d + 1e-12)

                Q1[:, i] = q
                Err1[:, i] = err

                if i + 1 < count:
                    W1[:, i + 1 :] -= err.unsqueeze(1) @ U1[i, i + 1 :].unsqueeze(0)

            W[:, c_start:c_end] = Q1
            losses[:, c_start:c_end] = Losses1 / 2.0

            if c_end < columns:
                W[:, c_end:] -= Err1 @ U[c_start:c_end, c_end:]

        self.layer.weight.data.copy_(W.to(dtype=self.layer.weight.dtype))
        return self.layer.weight.data


# --------------------------------------------------------------------------------------
# End-to-end helpers
# --------------------------------------------------------------------------------------


def iter_named_linear_layers(
    model: nn.Module,
    module_filter: Optional[Callable[[str, nn.Linear], bool]] = None,
) -> List[Tuple[str, nn.Linear]]:
    layers: List[Tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if module_filter is None or module_filter(name, module):
                layers.append((name, module))
    return layers


@torch.no_grad()
def collect_layer_stats(
    model: nn.Module,
    layer: nn.Linear,
    calibration_loader: Iterable[Any],
    *,
    device: torch.device | str,
    collect_wanda: bool,
    collect_hessian: bool,
    max_batches: Optional[int] = None,
) -> LayerStats:
    model.eval()
    device = torch.device(device)
    collector = LinearCalibrationCollector(
        layer,
        collect_wanda=collect_wanda,
        collect_hessian=collect_hessian,
    )

    def hook(_: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
        if not inputs:
            return
        collector.add_batch(inputs[0])

    handle = layer.register_forward_hook(hook)
    try:
        for batch_idx, batch in enumerate(calibration_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = _move_to_device(batch, device)
            _run_model_forward(model, batch)
    finally:
        handle.remove()

    return collector.as_stats()


@torch.no_grad()
def prune_model_sequentially(
    model: nn.Module,
    calibration_loader: Iterable[Any],
    *,
    method: str,
    sparsity: float,
    device: torch.device | str,
    module_filter: Optional[Callable[[str, nn.Linear], bool]] = None,
    max_batches: Optional[int] = None,
    prunen: int = 0,
    prunem: int = 0,
    percdamp: float = 0.01,
    blocksize: int = 128,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Sequentially prune linear layers using updated activations from previously pruned layers.

    This mirrors the paper's important idea that after pruning earlier layers, later layers should
    receive activations from the already-pruned network.
    """
    method = method.lower()
    if method not in {"magnitude", "wanda", "sparsegpt"}:
        raise ValueError("method must be one of: 'magnitude', 'wanda', 'sparsegpt'")

    target_layers = iter_named_linear_layers(model, module_filter=module_filter)
    summary: Dict[str, Dict[str, float]] = {}

    for name, layer in target_layers:
        if verbose:
            print(f"Pruning {name} with {method}...")

        if method == "magnitude":
            pruner = MagnitudePruner(layer)
        elif method == "wanda":
            stats = collect_layer_stats(
                model,
                layer,
                calibration_loader,
                device=device,
                collect_wanda=True,
                collect_hessian=False,
                max_batches=max_batches,
            )
            pruner = WandaPruner(layer, stats)
        else:
            stats = collect_layer_stats(
                model,
                layer,
                calibration_loader,
                device=device,
                collect_wanda=False,
                collect_hessian=True,
                max_batches=max_batches,
            )
            pruner = SparseGPTPruner(layer, stats)

        if method == "sparsegpt":
            pruner.prune(
                sparsity,
                prunen=prunen,
                prunem=prunem,
                percdamp=percdamp,
                blocksize=blocksize,
            )
        else:
            pruner.prune(sparsity, prunen=prunen, prunem=prunem)

        zeros, total = _count_zeros(layer.weight.data)
        summary[name] = {
            "zeros": float(zeros),
            "total": float(total),
            "sparsity": zeros / max(total, 1),
        }

    return summary


# --------------------------------------------------------------------------------------
# Minimal example
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    class TinyMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 4),
            )

        def forward(self, x: Tensor) -> Tensor:
            return self.net(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    model = TinyMLP().to(device).eval()
    calibration_loader = [torch.randn(8, 16) for _ in range(8)]

    summary = prune_model_sequentially(
        model,
        calibration_loader,
        method="wanda",
        sparsity=0.5,
        device=device,
        verbose=True,
    )
    print(summary)
