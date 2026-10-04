from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Callable, Iterable

import numpy as np
import pandas as pd

MULTI_SPACE_RE = re.compile(r"\s+")


def normalize_vietnamese_text(text: str) -> str:
    """Light normalization for Vietnamese sequence analysis."""
    t = unicodedata.normalize("NFC", str(text or ""))
    t = t.replace("\u00a0", " ")
    return MULTI_SPACE_RE.sub(" ", t).strip()


def whitespace_tokenize(text: str) -> list[str]:
    return normalize_vietnamese_text(text).lower().split()


def underthesea_tokenize(text: str) -> list[str]:
    try:
        from underthesea import word_tokenize
    except ImportError as exc:
        raise ImportError(
            "Vietnamese word segmentation requires underthesea. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    normalized = normalize_vietnamese_text(text).lower()
    segmented = word_tokenize(normalized, format="text")
    return segmented.split()


def get_tokenizer(mode: str) -> Callable[[str], list[str]]:
    if mode == "whitespace":
        return whitespace_tokenize
    if mode == "underthesea":
        return underthesea_tokenize
    raise ValueError(f"Unsupported tokenizer mode: {mode}")


def build_token_vocab(
    texts: Iterable[str],
    tokenizer: Callable[[str], list[str]],
    max_vocab_size: int,
) -> dict[str, int]:
    counter = Counter()
    for text in texts:
        counter.update(tokenizer(text))

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, _ in counter.most_common(max(max_vocab_size - 2, 0)):
        vocab[token] = len(vocab)
    return vocab


def audit_sequence_texts(
    texts: Iterable[str],
    vocab: dict[str, int],
    tokenizer: Callable[[str], list[str]],
    max_len: int,
) -> dict:
    orig_lens = []
    seq_lens = []
    total_tokens = 0
    total_unk = 0
    truncated = 0

    for text in texts:
        tokens = tokenizer(text)
        orig_len = len(tokens)
        ids = [vocab.get(tok, 1) for tok in tokens]

        orig_lens.append(orig_len)
        seq_len = min(orig_len, max_len)
        seq_lens.append(seq_len)
        total_tokens += orig_len
        total_unk += sum(1 for token_id in ids if token_id == 1)
        truncated += int(orig_len > max_len)

    orig = np.asarray(orig_lens, dtype=float)
    seq = np.asarray(seq_lens, dtype=float)
    n = len(orig_lens)

    return {
        "n_rows": int(n),
        "vocab_size": int(len(vocab)),
        "max_len": int(max_len),
        "orig_len_median": float(np.median(orig)) if n else 0.0,
        "orig_len_p95": float(np.percentile(orig, 95)) if n else 0.0,
        "truncation_rate": float(truncated / n) if n else 0.0,
        "unk_rate": float(total_unk / max(total_tokens, 1)),
        "avg_pad_ratio": float(np.mean((max_len - seq) / max_len)) if n else 0.0,
    }


def token_preview(
    texts: Iterable[str],
    tokenizer: Callable[[str], list[str]],
    n_examples: int = 5,
) -> pd.DataFrame:
    rows = []
    for idx, text in enumerate(texts):
        if idx >= n_examples:
            break
        normalized = normalize_vietnamese_text(text)
        tokens = tokenizer(normalized)
        rows.append(
            {
                "example_id": idx,
                "text": normalized,
                "tokens": " | ".join(tokens[:80]),
                "n_tokens": len(tokens),
            }
        )
    return pd.DataFrame(rows)
