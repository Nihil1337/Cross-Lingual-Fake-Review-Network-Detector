# Data Audit

Audit thực hiện trên các file hiện có trong thư mục `data/`.

## Snapshot

| Dataset/file group | Rows | Columns chính | Nhận xét |
|---|---:|---|---|
| Amazon Reviews Multi/train.csv | 1,200,000 | review/product/reviewer, stars, text, language, category | 6 ngôn ngữ, mỗi ngôn ngữ 200,000 dòng; rating cân bằng |
| Amazon Reviews Multi/validation.csv | 30,000 | như train | 5,000 dòng mỗi ngôn ngữ |
| Amazon Reviews Multi/test.csv | 30,000 | như train | 5,000 dòng mỗi ngôn ngữ |
| UIT-ViSFD/Train.csv | 7,786 | comment, n_star, date_time, label | tiếng Việt; aspect-based sentiment |
| UIT-ViSFD/Dev.csv | 1,112 | comment, n_star, date_time, label | tiếng Việt; aspect-based sentiment |
| UIT-ViSFD/Test.csv | 2,224 | comment, n_star, date_time, label | tiếng Việt; aspect-based sentiment |

## Amazon Reviews Multi

- Languages: `de`, `en`, `es`, `fr`, `ja`, `zh`.
- Có các trường hữu ích cho retrieval/graph: `review_id`, `product_id`, `reviewer_id`, `stars`, `review_body`, `review_title`, `language`, `product_category`.
- `review_id` không trùng giữa train, validation và test.
- `product_id` và `reviewer_id` có giao nhau giữa các split. Điều này phù hợp với một số bài toán in-domain nhưng có thể tạo đánh giá lạc quan nếu model học tín hiệu product/reviewer.
- Train có 4 dòng thiếu `review_title`; các file validation/test không có missing được phát hiện ở lần kiểm tra này.
- Dataset hiện không có cột ngày đăng. Không thể kết luận về xu hướng theo thời điểm trên Amazon nếu chưa bổ sung timestamp.
- Các ID mẫu được namespace theo ngôn ngữ, ví dụ `product_de_...`. Không nên coi hai ID khác prefix là cùng một sản phẩm nếu chưa có mapping sản phẩm xuyên thị trường.

## UIT-ViSFD

UIT-ViSFD là Vietnamese Smartphone Feedback Dataset cho aspect-based sentiment analysis, gồm 11,122 feedback smartphone được chia train/dev/test. Nhãn có dạng như `{CAMERA#Positive}` hoặc `{BATTERY#Negative}`.

Dataset này:

- không có nhãn fake/spam;
- không có `product_id` hoặc `reviewer_id` tương thích với Amazon Reviews Multi;
- có `date_time`, `n_star` và nhãn aspect/polarity;
- phù hợp để kiểm tra robustness tiếng Việt, tạo hard negative theo cùng aspect/rating, hoặc minh họa phân tích domain;
- không nên được dùng để báo cáo Precision/Recall của fake-review detection.

Nguồn tham khảo: <https://github.com/LuongPhan/UIT-ViSFD>

## Những thứ nên bổ sung

### Cần bổ sung trước khi đánh giá nghiêm túc

1. **Data manifest:** nguồn tải, license, ngày tải, schema, encoding và checksum cho từng file.
2. **Nguồn có timestamp cho Amazon:** nếu muốn giữ câu hỏi “review tập trung theo thời điểm”. Nếu chưa có nguồn này, tạm bỏ temporal feature khỏi kết luận chính.
3. **Split theo nhóm:** tạo ít nhất một evaluation track không để product hoặc reviewer xuất hiện đồng thời ở train/test.
4. **Metadata cho synthetic pairs:** lưu source review, ngôn ngữ đích, phương pháp biến đổi, mức độ paraphrase và nhãn positive/negative.
5. **Vietnamese cross-lingual plan:** ghi rõ cặp Amazon–Việt là synthetic, hoặc bổ sung dataset tiếng Việt có product/entity mapping nếu muốn phân tích dữ liệu thật xuyên ngôn ngữ.

### Nên bổ sung để project dễ chạy lại

- một sample nhỏ trong `data/sample/` để test parser và API mà không cần tải dữ liệu lớn;
- `data/processed/` cho normalized text, feature table và synthetic metadata;
- `artifacts/` cho embeddings, ANN index, model và graph summary;
- `.gitignore` loại raw data, embeddings, index và model lớn khỏi version control;
- file cấu hình ghi rõ subset, model, batch size, top-k, threshold và random seed.

## Khuyến nghị triển khai ngay

Có thể bắt đầu POC mà chưa cần thêm dataset mới:

1. dùng Amazon train/validation/test cho multilingual embedding và ANN retrieval;
2. tạo synthetic translation/paraphrase pairs trong 2–3 ngôn ngữ trước;
3. dùng product/category/rating làm context, chưa dùng temporal feature;
4. dùng UIT-ViSFD riêng cho kiểm tra tiếng Việt và aspect-aware hard negative;
5. chỉ mở rộng sang nguồn có timestamp sau khi pipeline pair-level đã ổn định.

Như vậy, dữ liệu hiện có đủ để bắt đầu xây pipeline. Phần còn thiếu lớn nhất không phải thêm thật nhiều dữ liệu, mà là timestamp cho temporal analysis, metadata synthetic và protocol đánh giá chống leakage.

