from datasets import load_dataset


def load_wikitext_validation():
    """
    Load the WikiText-2 validation split.
    """
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join(dataset["text"])
    return text


def tokenize_wikitext(
    tokenizer,
    sequence_length=2048,
):
    """
    Tokenize WikiText validation into a single tensor stream.
    """
    text = load_wikitext_validation()

    tokens = tokenizer(
        text,
        return_tensors="pt",
    ).input_ids

    if tokens.shape[1] < sequence_length:
        raise ValueError("WikiText dataset too short for evaluation.")

    return tokens