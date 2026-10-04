from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.vietnamese_text import (
    audit_sequence_texts,
    build_token_vocab,
    get_tokenizer,
    normalize_vietnamese_text,
    token_preview,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--text_col", default="content")
    ap.add_argument("--label_col", default="category")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_rows", type=int, default=None)
    ap.add_argument("--vocab_size", type=int, default=20000)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument(
        "--modes",
        nargs="+",
        default=["whitespace", "underthesea"],
        choices=["whitespace", "underthesea"],
    )
    ap.add_argument("--output_dir", default="outputs/vietnews")
    return ap.parse_args()


def can_stratify(labels: pd.Series) -> bool:
    if labels.nunique() < 2:
        return False
    return bool((labels.value_counts() >= 2).all())


def render_audit(path: Path, mode: str, train_audit: dict, test_audit: dict) -> None:
    text = f"""# VietNewsSense Sequence Audit — {mode}

## Train-side vocabulary
- vocab_size: {train_audit['vocab_size']}
- max_len: {train_audit['max_len']}
- train_orig_len_median: {train_audit['orig_len_median']:.2f}
- train_orig_len_p95: {train_audit['orig_len_p95']:.2f}

## Held-out sequence behavior
- n_test: {test_audit['n_rows']}
- test_orig_len_median: {test_audit['orig_len_median']:.2f}
- test_orig_len_p95: {test_audit['orig_len_p95']:.2f}
- test_unk_rate: {test_audit['unk_rate']:.4f}
- test_truncation_rate: {test_audit['truncation_rate']:.4f}
- test_avg_pad_ratio: {test_audit['avg_pad_ratio']:.4f}

## Gợi ý đọc kết quả
- UNK rate cao: vocabulary hoặc tokenization chưa bao phủ tốt dữ liệu held-out.
- Truncation rate cao: max_len có thể quá nhỏ.
- Padding ratio cao: max_len có thể lớn hơn cần thiết.
- So sánh hai tokenizer bằng cùng split, vocab_size và max_len.
"""
    path.write_text(text, encoding="utf-8")


def render_comparison(path: Path, result_df: pd.DataFrame) -> None:
    cols = [
        "tokenizer",
        "vocab_size",
        "test_len_median",
        "test_len_p95",
        "test_unk_rate",
        "test_truncation_rate",
        "test_avg_pad_ratio",
    ]
    view = result_df[cols].copy()
    markdown = view.to_markdown(index=False, floatfmt=".4f")
    text = f"""# VietNewsSense — Sequence Transfer Comparison

Cùng dataset, cùng split, cùng vocab limit và cùng max_len; chỉ thay tokenization.

{markdown}

## Reflection prompts
1. Tokenization làm thay đổi vocabulary và sequence length như thế nào?
2. Với cùng max_len, cách nào tạo nhiều truncation hoặc padding hơn?
3. UNK rate held-out thay đổi ra sao?
4. Những thay đổi này có thể ảnh hưởng Embedding + RNN như thế nào?

## Lưu ý
Đây là sequence transfer check, không phải cuộc thi accuracy giữa hai tokenizer.
Không cần train thêm một RNN đầy đủ trên VietNewsSense trong Lab 3.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_path)
    if args.text_col not in df.columns:
        raise ValueError(f"Missing text column: {args.text_col}. Available: {list(df.columns)}")
    if args.label_col not in df.columns:
        raise ValueError(f"Missing label column: {args.label_col}. Available: {list(df.columns)}")

    df = df.copy()
    df["text"] = df[args.text_col].fillna("").astype(str).map(normalize_vietnamese_text)
    df["label"] = df[args.label_col].astype(str)

    if args.max_rows is not None and args.max_rows < len(df):
        df, _ = train_test_split(
            df,
            train_size=int(args.max_rows),
            random_state=args.seed,
            stratify=df["label"] if can_stratify(df["label"]) else None,
        )
        df = df.reset_index(drop=True)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=args.seed,
        stratify=df["label"] if can_stratify(df["label"]) else None,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    rows = []
    previews = []

    for mode in args.modes:
        tokenizer = get_tokenizer(mode)
        vocab = build_token_vocab(
            train_df["text"].tolist(),
            tokenizer=tokenizer,
            max_vocab_size=args.vocab_size,
        )
        train_audit = audit_sequence_texts(
            train_df["text"].tolist(),
            vocab=vocab,
            tokenizer=tokenizer,
            max_len=args.max_len,
        )
        test_audit = audit_sequence_texts(
            test_df["text"].tolist(),
            vocab=vocab,
            tokenizer=tokenizer,
            max_len=args.max_len,
        )

        render_audit(
            out_dir / f"{mode}_sequence_audit.md",
            mode,
            train_audit,
            test_audit,
        )

        preview_df = token_preview(
            test_df["text"].tolist(),
            tokenizer=tokenizer,
            n_examples=5,
        )
        preview_df.insert(0, "tokenizer", mode)
        previews.append(preview_df)

        rows.append(
            {
                "tokenizer": mode,
                "n_total": int(len(df)),
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "n_classes": int(df["label"].nunique()),
                "vocab_size": train_audit["vocab_size"],
                "max_len": args.max_len,
                "test_len_median": test_audit["orig_len_median"],
                "test_len_p95": test_audit["orig_len_p95"],
                "test_unk_rate": test_audit["unk_rate"],
                "test_truncation_rate": test_audit["truncation_rate"],
                "test_avg_pad_ratio": test_audit["avg_pad_ratio"],
            }
        )

    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_dir / "transfer_comparison.csv", index=False)
    render_comparison(out_dir / "transfer_comparison.md", result_df)

    if previews:
        pd.concat(previews, ignore_index=True).to_csv(
            out_dir / "token_examples.csv",
            index=False,
        )

    print(f"DONE. VietNewsSense sequence audit: {out_dir}")


if __name__ == "__main__":
    main()
