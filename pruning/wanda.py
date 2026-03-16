import torch
from collections import defaultdict


class WandaPruner:
    """
    Implements the WANDA pruning criterion:

        score = |W_ij| * ||x_j||_2

    where ||x_j||_2 is the accumulated L2 norm of the input activation channel
    over calibration samples.
    """

    def __init__(self, model, sparsity):
        self.model = model
        self.sparsity = sparsity
        self.input_norms = defaultdict(lambda: None)
        self.handles = []

    def _hook_fn(self, name):
        def hook(module, inp, output):
            x = inp[0]

            # Flatten [batch, seq, hidden] -> [tokens, hidden]
            x = x.reshape(-1, x.shape[-1]).detach()

            # Per-input-channel L2 norm
            norm = torch.norm(x, p=2, dim=0)

            if self.input_norms[name] is None:
                self.input_norms[name] = norm
            else:
                self.input_norms[name] += norm

        return hook

    def register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                handle = module.register_forward_hook(self._hook_fn(name))
                self.handles.append(handle)

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def collect_activation_stats(self, dataloader, device):
        self.register_hooks()
        self.model.eval()

        with torch.no_grad():
            for i, batch in enumerate(dataloader, start=1):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                    self.model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    self.model(input_ids=input_ids)

                print(
                    f"[wanda] collected calibration batch {i}",
                    end="\r",
                    flush=True,
                )

        print()
        self.remove_hooks()

    def prune(self):
        masks = {}

        for name, module in self.model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue

            W = module.weight.data
            act_norm = self.input_norms[name].to(W.device, dtype=W.dtype)

            # score shape: [out_features, in_features]
            score = W.abs() * act_norm.unsqueeze(0)

            _, in_features = score.shape
            k = int(in_features * self.sparsity)

            if k <= 0:
                masks[name] = True
                continue

            prune_idx = torch.topk(score, k=k, dim=1, largest=False).indices
            mask = torch.zeros_like(W, dtype=torch.bool)
            mask.scatter_(1, prune_idx, True)

            module.weight.data.masked_fill_(mask, 0)
            masks[name] = True

        return masks