import torch


def magnitude_prune_linear_layer(layer, sparsity):
    """
    Unstructured row-wise magnitude pruning for a Linear layer.
    Vectorized implementation.
    """
    W = layer.weight.data
    _, in_features = W.shape
    k = int(in_features * sparsity)

    if k <= 0:
        return True

    scores = W.abs()
    prune_idx = torch.topk(scores, k=k, dim=1, largest=False).indices

    mask = torch.zeros_like(W, dtype=torch.bool)
    mask.scatter_(1, prune_idx, True)

    layer.weight.data.masked_fill_(mask, 0)
    return True


def magnitude_prune_model(model, sparsity):
    """
    Apply magnitude pruning to all Linear layers in the model.
    """
    masks = {}

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            mask = magnitude_prune_linear_layer(module, sparsity)
            masks[name] = True

    return masks