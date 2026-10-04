# CSC4007 — Lab 3 Analysis Report
## RNN + W&B + VietNewsSense Sequence Transfer Check

## 1. Thông tin sinh viên
- Họ và tên:
- Mã sinh viên:
- Lớp:
- Repo GitHub:
- W&B project:
- Tên run tốt nhất:

## 2. Mục tiêu thí nghiệm
Viết 3–5 dòng:
- Lab 3 khác Lab 2 ở điểm nào?
- Vì sao chuyển từ BoW/TF-IDF sang token sequence?
- RNN có thể khai thác điều gì mà baseline tuyến tính không biểu diễn trực tiếp?

# PART A — IMDB CORE LAB

## 3. Sequence audit
Dựa trên `outputs/logs/sequence_audit.md`, nêu ít nhất 3 nhận xét có số liệu.

1.
2.
3.

Gợi ý:
- median/p95 length?
- `max_len` hợp lý không?
- truncation rate?
- padding ratio?
- UNK rate?

## 4. Thiết lập mô hình và huấn luyện
- vocab_size:
- max_len:
- embed_dim:
- hidden_dim:
- batch_size:
- epochs:
- learning rate:
- dropout:
- seed:
- early stopping patience:
- wandb_mode:

Giải thích vì sao chọn cấu hình này.

## 5. Baseline ML vs RNN

| Mô hình | Representation | Accuracy | Macro-F1 | Ghi chú |
|---|---|---:|---:|---|
| Baseline ML — Lab 2 | BoW/TF-IDF |  |  |  |
| RNN — Lab 3 | token sequence + embedding |  |  |  |

Trả lời:
- RNN có tốt hơn baseline không?
- Nếu tốt hơn, chênh lệch có đáng kể không?
- Nếu chưa tốt hơn, nguyên nhân hợp lý là gì?
- Vai trò của word order thể hiện ở đâu?

## 6. Learning curves và W&B
Đính kèm:
- `loss_curve.png`
- `metric_curve.png`
- hoặc ảnh W&B

Trả lời:
- Epoch tốt nhất?
- Có overfitting không?
- Run nào tốt hơn?
- W&B giúp quan sát điều gì?

## 7. Error analysis — ít nhất 10 lỗi IMDB

| ID | True | Pred | Nhóm lỗi | Vì sao sai? | Hướng cải thiện |
|---|---|---|---|---|---|
|  |  |  |  |  |  |

Gợi ý:
- negation
- mixed sentiment
- long review
- sarcasm/irony
- confident but wrong
- long-range context

# PART B — VIETNEWSSENSE SEQUENCE TRANSFER CHECK

## 8. Thiết lập transfer check

- data_path:
- text_col:
- label_col:
- seed:
- vocab_size:
- max_len:
- modes: `whitespace`, `underthesea`

Xác nhận: hai tokenizer sử dụng cùng split, vocab limit và max_len.

## 9. Kết quả sequence transfer

Điền từ `outputs/vietnews/transfer_comparison.csv`.

| Tokenizer | Vocab size | Median len | P95 len | UNK rate | Truncation rate | Padding ratio |
|---|---:|---:|---:|---:|---:|---:|
| whitespace |  |  |  |  |  |  |
| underthesea |  |  |  |  |  |  |

### Nhận xét
1. Vocabulary thay đổi như thế nào?
2. Sequence length thay đổi như thế nào?
3. Với cùng `max_len`, truncation/padding thay đổi ra sao?
4. UNK rate held-out thay đổi ra sao?

## 10. Token example

Chọn ít nhất 2 ví dụ từ `token_examples.csv`.

### Ví dụ 1
- Raw text:
- Whitespace tokens:
- Underthesea tokens:
- Nhận xét:

### Ví dụ 2
- Raw text:
- Whitespace tokens:
- Underthesea tokens:
- Nhận xét:

## 11. Vietnamese Transfer Reflection

Trả lời 5–7 dòng:

> Những thay đổi về vocabulary, sequence length, UNK, truncation và padding có thể ảnh hưởng Embedding + RNN như thế nào?

Không được kết luận tokenizer nào tạo accuracy cao hơn vì Part B không train hai RNN để so accuracy.

## 12. Bài học rút ra

Viết 5–7 dòng:
- ưu/nhược điểm RNN;
- sequence length;
- validation/early stopping;
- learning curves;
- vì sao tokenization là một phần của thiết kế mô hình chuỗi;
- điểm muốn kiểm chứng tiếp ở Lab 4 LSTM.
