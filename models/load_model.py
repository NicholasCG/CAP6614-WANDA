import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    model_name="meta-llama/Llama-2-7b-hf",
    dtype="float16",
    device=None,
):
    """
    Load a HuggingFace causal LM and tokenizer.
    """

    if dtype in ["float16", "fp16"]:
        torch_dtype = torch.float16
    elif dtype in ["bfloat16", "bf16"]:
        torch_dtype = torch.bfloat16
    elif dtype in ["float32", "fp32"]:
        torch_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    return model, tokenizer, device