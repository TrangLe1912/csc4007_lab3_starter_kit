# data/raw

- Lab 3 Core mặc định dùng **IMDB** từ Hugging Face (`stanfordnlp/imdb`).
- `sample_imdb_tiny.csv` dùng cho smoke test cục bộ.
- `sample_vietnews_tiny.csv` là dữ liệu tiếng Việt tổng hợp nhỏ để CI kiểm tra sequence transfer.
- Hai file sample chỉ phục vụ kiểm tra code, **không dùng để báo cáo kết quả học thuật**.
- Với VietNewsSense thật, đặt file tại `data/raw/vietnewssense.csv` hoặc truyền đường dẫn khác qua `--data_path`.
- Schema transfer check mặc định:
  - `content`: nội dung bài báo
  - `category`: nhãn chủ đề
