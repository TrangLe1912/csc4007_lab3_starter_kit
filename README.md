# CSC4007 — Lab 3: Sequence Models with RNN + Weights & Biases
## IMDB Core Lab + VietNewsSense Sequence Transfer Check

Lab 3 tiếp tục trực tiếp từ Lab 2:

`BoW / TF-IDF baseline → token sequence → Embedding → RNN → evaluate → error analysis`

Thiết kế bài lab gồm hai phần:

- **Part A — Core Lab (IMDB):** huấn luyện Embedding + RNN, theo dõi bằng W&B và so sánh với baseline Lab 2.
- **Part B — Vietnamese Sequence Transfer Check (VietNewsSense):** không train thêm một RNN đầy đủ; chỉ kiểm chứng tokenization làm thay đổi vocabulary, sequence length, UNK, truncation và padding như thế nào.

Mục tiêu là giữ IMDB làm trục so sánh kiến trúc, đồng thời tiếp tục case study tiếng Việt theo đúng câu hỏi của Bài 3: **pipeline có thể khái quát, nhưng representation của chuỗi phụ thuộc ngôn ngữ và tokenization.**

## Mục tiêu

Sau Lab 3, sinh viên cần:

1. Biểu diễn văn bản thành chuỗi token thay vì BoW/TF-IDF.
2. Xây dựng và huấn luyện mô hình **Embedding + RNN** trên IMDB.
3. Hiểu vocabulary, padding, `max_len`, embedding, hidden state, dropout và early stopping.
4. Sử dụng W&B để theo dõi learning curves và so sánh run.
5. So sánh baseline ML của Lab 2 với RNN của Lab 3.
6. Phân tích ít nhất 10 lỗi IMDB.
7. Thực hiện **VietNewsSense Sequence Transfer Check** với hai cách tokenization.

## Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Nếu dùng W&B online:

```bash
wandb login
```

---

# Part A — Core Lab: IMDB

## Chạy mô hình chính

```bash
python run_lab3.py \
  --dataset imdb \
  --seed 42 \
  --vocab_size 20000 \
  --max_len 256 \
  --embed_dim 128 \
  --hidden_dim 128 \
  --batch_size 64 \
  --epochs 6 \
  --lr 1e-3 \
  --dropout 0.3 \
  --use_wandb
```

IMDB dùng split gốc; validation được tách từ train, test split gốc chỉ dùng cho đánh giá cuối.

Sinh viên cần thử **ít nhất 2 cấu hình**, có thể thay:
- `max_len`
- `hidden_dim`
- `dropout`
- `lr`
- `batch_size`
- `patience`

## Nối với baseline Lab 2

```bash
python run_lab3.py \
  --dataset imdb \
  --baseline_metrics_path /path/to/lab2/outputs/imdb_baseline/metrics/metrics_summary.json
```

Repo sẽ tạo `outputs/metrics/baseline_vs_rnn.csv`.

## Smoke test cục bộ

```bash
python run_lab3.py \
  --dataset local_csv \
  --data_path data/raw/sample_imdb_tiny.csv \
  --epochs 2 \
  --batch_size 8 \
  --max_len 64 \
  --vocab_size 2000 \
  --embed_dim 32 \
  --hidden_dim 32 \
  --wandb_mode offline \
  --use_wandb
```

---

# Part B — VietNewsSense Sequence Transfer Check

## Câu hỏi thực nghiệm

> Khi chuyển từ tiếng Anh sang tin tức tiếng Việt, tokenization làm thay đổi dữ liệu đầu vào của RNN như thế nào?

Trong phần này **không train thêm một RNN đầy đủ**. Sinh viên chỉ giữ:
- cùng dataset;
- cùng split;
- cùng `vocab_size`;
- cùng `max_len`;

và thay đổi duy nhất cách tokenization:

1. `whitespace`
2. `underthesea`

## Chạy transfer check

Giả sử VietNewsSense có cột:
- `content`: nội dung
- `category`: nhãn chủ đề

```bash
python run_vietnews_sequence_audit.py \
  --data_path data/raw/vietnewssense.csv \
  --text_col content \
  --label_col category \
  --seed 42 \
  --vocab_size 20000 \
  --max_len 256 \
  --modes whitespace underthesea \
  --output_dir outputs/vietnews
```

## Các chỉ số cần so sánh

- `vocab_size`
- median sequence length
- p95 sequence length
- held-out `UNK rate`
- `truncation rate`
- `average padding ratio`

Output:

```text
outputs/vietnews/
├── whitespace_sequence_audit.md
├── underthesea_sequence_audit.md
├── transfer_comparison.csv
├── transfer_comparison.md
└── token_examples.csv
```

### Không được kết luận

Không dùng Part B để kết luận:
- tokenizer nào “tốt hơn” chỉ từ sequence statistics;
- tiếng Việt “khó hơn” tiếng Anh;
- word segmentation luôn giúp RNN.

Part B chỉ trả lời: **tokenization thay đổi hình dạng sequence đầu vào như thế nào và điều đó có thể ảnh hưởng RNN ra sao.**

---

# Output Part A

Sau khi chạy `run_lab3.py`, repo sinh:

- `outputs/logs/sequence_audit.md`
- `outputs/metrics/epoch_history.csv`
- `outputs/metrics/metrics_summary.json`
- `outputs/metrics/metrics_summary.md`
- `outputs/metrics/baseline_vs_rnn.csv`
- `outputs/figures/loss_curve.png`
- `outputs/figures/metric_curve.png`
- `outputs/figures/confusion_matrix.png`
- `outputs/error_analysis/error_analysis.csv`
- `outputs/error_analysis/error_analysis_summary.md`
- `outputs/models/best_model.pt`
- `outputs/predictions/test_predictions.csv`
- `outputs/logs/run_summary.json`

# Yêu cầu nộp bài

Sinh viên cần:

1. Chạy Embedding + RNN trên IMDB.
2. Log ít nhất một run hoàn chỉnh bằng W&B.
3. Thử ít nhất 2 cấu hình.
4. So sánh baseline Lab 2 với RNN Lab 3.
5. Phân tích ít nhất 10 mẫu sai IMDB.
6. Chạy VietNewsSense Sequence Transfer Check với `whitespace` và `underthesea`.
7. Hoàn thành `reports/analysis_report.md`.

# Cấu trúc chính

```text
csc4007_lab3_starter_kit/
├── .github/workflows/
├── data/raw/
├── reports/
│   ├── analysis_report.md
│   └── rubric.md
├── run_lab3.py
├── run_vietnews_sequence_audit.py
├── requirements.txt
└── src/
    ├── data.py
    ├── model.py
    ├── sequence_audit.py
    ├── vietnamese_text.py
    └── ...
```

## CI

Repo kiểm tra:
- đường chạy IMDB/RNN;
- smoke test local IMDB;
- VietNewsSense sequence audit bằng dữ liệu Việt nhỏ tổng hợp.

CI chỉ kiểm tra code/artefact, không thay thế phân tích học thuật.
