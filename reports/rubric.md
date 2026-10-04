# CSC4007 — Lab 3 Rubric
## RNN + W&B + VietNewsSense Sequence Transfer

Tổng điểm: **10 điểm**

## 1. Repo chạy được và artefact đầy đủ — 1.5 điểm
- **1.5**: Core RNN và transfer check đều chạy; artefact đầy đủ, đường dẫn rõ.
- **0.75**: Chạy được nhưng thiếu artefact hoặc chỉ hoàn thành một phần.
- **0.0**: Repo không chạy được.

## 2. Mô hình RNN và quy trình huấn luyện — 2.0 điểm
- **2.0**: Embedding + RNN đúng; validation; seed; early stopping/kiểm soát overfit hợp lý.
- **1.0**: Chạy được nhưng quy trình chưa chặt.
- **0.0**: Sai logic hoặc không train được.

## 3. Sử dụng W&B — 1.0 điểm
- **1.0**: Log hyperparameters/metrics và dùng để so sánh run.
- **0.5**: Có dùng nhưng sơ sài.
- **0.0**: Không có bằng chứng.

## 4. Baseline ML vs RNN — 1.5 điểm
- **1.5**: So sánh đúng Lab 2 vs Lab 3, có số liệu và diễn giải hợp lý.
- **0.75**: Có bảng nhưng phân tích còn mỏng.
- **0.0**: Không so sánh hoặc so sánh sai.

## 5. Learning curves và diễn giải — 1.5 điểm
- **1.5**: Đọc đúng loss/metric curves, epoch tốt nhất, over/underfitting.
- **0.75**: Có hình nhưng diễn giải hạn chế.
- **0.0**: Thiếu.

## 6. Error analysis IMDB — 1.0 điểm
- **1.0**: Phân tích >=10 mẫu, nhóm lỗi và đề xuất cải thiện.
- **0.5**: Có nhưng chưa đủ/chưa sâu.
- **0.0**: Không có.

## 7. VietNewsSense Sequence Transfer Check — 1.5 điểm
- **1.5**: Chạy cả whitespace + underthesea; bảng so sánh đầy đủ; dùng token examples; reflection đúng phạm vi.
- **0.75**: Có chạy nhưng thiếu một phần metric/reflection.
- **0.0**: Không thực hiện hoặc kết luận sai bản chất thực nghiệm.

## Ghi chú trừ điểm
- So sánh accuracy IMDB và VietNewsSense để kết luận ngôn ngữ nào khó hơn.
- Kết luận underthesea “tốt hơn” chỉ từ sequence statistics.
- Không giữ cùng split/vocab_size/max_len khi so tokenization.
- Chỉ báo cáo accuracy, bỏ macro-F1.
- Không phân biệt train/validation/test.
