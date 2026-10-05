# Cross-Lingual Fake Review Network Detector

## 1. Tóm tắt ý tưởng

Xây dựng một hệ thống phát hiện các review sản phẩm có khả năng là bản dịch hoặc diễn đạt lại của một review gốc khác, kể cả khi chúng được viết bằng các ngôn ngữ khác nhau. Hệ thống kết hợp:

- multilingual semantic embeddings để tìm các cặp review có nội dung tương đồng;
- thông tin ngữ cảnh như sản phẩm, category, rating và thời gian đăng;
- một lớp verification để giảm false positive;
- graph analysis để tìm các cụm review bất thường theo sản phẩm và thời điểm;
- API và dashboard để tra cứu, giải thích và trình bày kết quả.

Mục tiêu là phát hiện **mẫu nội dung trùng lặp ngữ nghĩa xuyên ngôn ngữ**, không phải khẳng định danh tính của reviewer hay kết luận pháp lý rằng một reviewer cụ thể thuộc một review farm.

## 2. Câu hỏi nghiên cứu

Trong hàng triệu review sản phẩm ở nhiều ngôn ngữ:

1. Có bao nhiêu review là bản dịch hoặc diễn đạt lại của một review gốc khác?
2. Các cặp/cụm review này có tập trung bất thường ở cùng sản phẩm, category, rating hoặc khoảng thời gian nào không?
3. Multilingual embedding trực tiếp có hiệu quả hơn pipeline dịch về một ngôn ngữ trung gian rồi so sánh đơn ngữ không?
4. Các tín hiệu ngữ nghĩa, thời gian, rating và sản phẩm kết hợp với nhau đến mức nào trong việc xếp hạng các cặp đáng điều tra?

## 3. Đánh giá tính khả thi

### Kết luận

Đề tài **khả thi** ở quy mô đồ án hoặc portfolio full-stack data science. Kiến trúc có tính nghiên cứu, có phần machine learning, information retrieval, graph mining, backend và visualization. Tuy nhiên, phạm vi nên được triển khai theo từng tầng thay vì cố xử lý toàn bộ dữ liệu ngay từ đầu.

### Vì sao khả thi

- Bài toán candidate retrieval phù hợp với embedding đa ngôn ngữ và ANN index.
- Nhãn synthetic có thể tạo được có kiểm soát, giúp đo Precision/Recall trên một benchmark mà ta biết quan hệ giữa review gốc và review biến đổi.
- Amazon Reviews Multi có các thuộc tính hữu ích như ngôn ngữ, product/category, rating và ngày đăng; các trường thực tế cần được kiểm tra lại trong bước tải dữ liệu.
- Dữ liệu thật có thể dùng để chạy pipeline, lập graph và rút case study mà không cần tuyên bố nhãn fake review tuyệt đối.
- Có thể chứng minh pipeline bằng subset vài trăm nghìn review trước khi mở rộng; không cần encode hàng triệu review để chứng minh ý tưởng đúng.

### Điều kiện để đề tài đáng tin cậy

1. Phân biệt rõ ba loại kết quả:
   - **Định lượng có nhãn:** synthetic pairs và hard negatives.
   - **Đánh giá ablation:** multilingual embedding trực tiếp so với dịch rồi so sánh.
   - **Quan sát dữ liệu thật:** các cụm bất thường và case study, không phải ground truth fake review.
2. Tách train/validation/test theo product, template và phương pháp biến đổi để tránh cùng một review gốc xuất hiện ở nhiều tập.
3. Không dùng exact match làm phương pháp chính; exact/near-exact match chỉ là baseline và một feature phụ.
4. Đánh giá theo precision ở ngân sách điều tra thực tế, chẳng hạn Precision@100 hoặc Precision@1000, ngoài F1 và ROC-AUC.
5. Ghi lại phiên bản dataset, model, seed, tham số sinh synthetic và cấu hình ANN để kết quả có thể tái lập.

## 4. Ranh giới tuyên bố

Hệ thống được phép nói:

> Các review này có mức tương đồng ngữ nghĩa xuyên ngôn ngữ cao và có các tín hiệu đồng thời về sản phẩm, rating hoặc thời gian, nên đáng được ưu tiên điều tra.

Hệ thống không được tự động nói:

- reviewer chắc chắn là cùng một người;
- reviewer thuộc một tổ chức hoặc review farm cụ thể;
- review chắc chắn là fake;
- model đạt độ chính xác trên fake review thật nếu chưa có ground truth tương ứng.

Amazon Reviews Multi có thể ẩn danh reviewer. Nếu có reviewer ID, chỉ dùng ID đó như một tín hiệu cấu trúc trong dữ liệu và không suy luận danh tính ngoài dữ liệu.

## 5. Dữ liệu và chuẩn hóa

### Dữ liệu đề xuất

- **Amazon Reviews Multi:** nguồn chính để nghiên cứu cross-lingual similarity và chạy pipeline trên dữ liệu thật.
- **UIT-ViSFD:** Vietnamese Smartphone Feedback Dataset cho aspect-based sentiment analysis. Dataset này có nhãn aspect/polarity, không có nhãn fake/spam; dùng làm dữ liệu phụ để kiểm tra robustness trên tiếng Việt, phân tích domain và tạo hard negative có cùng aspect nhưng khác nội dung.
- **Synthetic benchmark:** dữ liệu bắt buộc để tạo nhãn pair-level có kiểm soát.

Trước khi dùng, cần kiểm tra schema, giấy phép, số lượng mẫu, trường ngôn ngữ, product ID, reviewer ID, rating và ngày đăng của từng dataset. Không giả định rằng hai dataset có thể nối trực tiếp thành các cặp cross-lingual thật.

### Audit dữ liệu hiện có trong project

| Dataset | Train | Validation/Dev | Test | Ngôn ngữ | Vai trò phù hợp |
|---|---:|---:|---:|---|---|
| Amazon Reviews Multi | 1,200,000 | 30,000 | 30,000 | de, en, es, fr, ja, zh | nguồn chính cho embedding, retrieval và graph |
| UIT-ViSFD | 7,786 | 1,112 | 2,224 | vi | domain/language robustness và aspect-aware hard negative |

Các phát hiện cần ghi nhớ:

- Amazon hiện có `review_id`, `product_id`, `reviewer_id`, `stars`, `review_body`, `review_title`, `language`, `product_category`, nhưng **không có `review_date`/`date_time`**. Vì vậy, phân tích review tập trung theo thời điểm chưa thể thực hiện trên bộ Amazon hiện tại; cần bổ sung một nguồn có timestamp hoặc coi temporal analysis là phần mở rộng tùy chọn.
- Amazon có 6 ngôn ngữ nhưng không có tiếng Việt. Các cặp Amazon–Việt hiện chỉ có thể tạo bằng dịch/paraphrase synthetic, trừ khi bổ sung một nguồn tiếng Việt có mapping sản phẩm hoặc mapping nội dung phù hợp.
- `review_id` không trùng giữa train/validation/test, nhưng `product_id` và `reviewer_id` có giao nhau giữa các split. Khi dùng product/reviewer làm feature, cần báo cáo cả random review split và group split theo product/reviewer.
- UIT-ViSFD có `comment`, `n_star`, `date_time`, `label`; `label` là các nhãn aspect/polarity như camera, battery, performance và positive/negative/neutral, không phải nhãn fake review.

### Chuẩn hóa

- chuẩn hóa Unicode và khoảng trắng nhưng giữ lại bản gốc để phục vụ giải thích;
- xử lý missing title/body và review quá ngắn;
- chuẩn hóa rating, timestamp, language code và category;
- loại hoặc đánh dấu các review trùng nguyên văn;
- lưu cả `raw_text` và `normalized_text`;
- tạo fingerprint để kiểm soát duplicate và phát hiện leakage.

EDA cần tập trung vào phân bố ngôn ngữ, category, rating, độ dài review, mật độ theo ngày và số review trên mỗi sản phẩm.

## 6. Sinh synthetic pairs

Mỗi review gốc có thể sinh nhiều mức biến đổi:

1. dịch máy sang một hoặc nhiều ngôn ngữ;
2. paraphrase nhẹ bằng cách thay đổi từ vựng và cấu trúc câu;
3. paraphrase trung bình với thay đổi thứ tự và cách diễn đạt;
4. giữ nguyên ý chính nhưng thay đổi đáng kể bề mặt văn bản;
5. tạo hard negative: cùng product/category hoặc cùng rating nhưng khác nội dung;
6. tạo negative ngẫu nhiên và negative gần theo embedding nhưng khác ý.

Mỗi mẫu nên có metadata về `source_review_id`, phương pháp biến đổi, mức độ biến đổi và ngôn ngữ. Không để cùng source review hoặc cùng template rơi vào cả train và test.

Synthetic data chỉ mô phỏng một số dạng sao chép có tổ chức. Nó không đại diện đầy đủ cho mọi cách review farm hoạt động, vì vậy kết quả trên synthetic không được trình bày như độ chính xác ngoài đời.

## 7. Pipeline kỹ thuật

### Bước 1: Candidate retrieval

1. Encode review bằng một hoặc vài multilingual sentence embedding models.
2. Xây ANN index bằng FAISS hoặc HNSW.
3. Tìm top-k ứng viên theo cosine similarity hoặc inner product.
4. Dùng blocking để giảm chi phí và false positive:
   - cùng product hoặc cùng category;
   - khoảng thời gian phù hợp;
   - rating giống hoặc gần nhau;
   - loại trừ self-match và duplicate đã biết.

Không nên lọc quá mạnh trước ANN vì có thể làm mất các cặp cross-lingual thật sự. Blocking nên được đo bằng candidate recall.

### Bước 2: Verification layer

Tạo feature cho từng cặp review, ví dụ:

- embedding cosine similarity;
- similarity theo title và body;
- độ dài và chênh lệch độ dài;
- lexical overlap sau khi chuẩn hóa hoặc dịch tham chiếu;
- cùng product/category hay không;
- cùng hoặc gần rating hay không;
- khoảng cách thời gian;
- cùng language hay khác language;
- chất lượng/độ tin cậy của embedding;
- số lượng review tương tự xung quanh cặp đó.

Model ban đầu có thể là Logistic Regression để dễ giải thích, sau đó so sánh với LightGBM/XGBoost hoặc một classifier tương đương. Nếu dùng threshold, cần hiệu chỉnh xác suất và báo cáo precision-recall theo nhiều ngưỡng.

### Bước 3: Graph construction

Có thể xây heterogeneous graph gồm:

- node review;
- node product;
- tùy chọn node reviewer ID ẩn danh, language và time bucket;
- edge review-review với trọng số từ similarity/verification score;
- edge review-product và review-reviewer nếu trường dữ liệu có sẵn.

Dùng threshold hoặc top-k edge để graph không quá dày. Leiden hoặc Louvain có thể dùng để tìm community. Khi diễn giải community, nên xem đồng thời phân bố product, rating, ngôn ngữ, timestamp và review text đại diện.

## 8. Thiết kế thực nghiệm

### Baseline

- exact match;
- TF-IDF/cosine trong cùng ngôn ngữ;
- character n-gram hoặc MinHash/Jaccard;
- dịch về một ngôn ngữ trung gian rồi so sánh đơn ngữ;
- multilingual embedding trực tiếp.

### Ablation

So sánh lần lượt:

1. embedding only;
2. embedding + product/category;
3. embedding + rating/time;
4. toàn bộ feature verification;
5. có và không có blocking;
6. multilingual embedding trực tiếp và translate-then-compare.

### Metrics

- Precision, Recall, F1 và PR-AUC ở pair-level;
- Recall@k trong candidate retrieval;
- Precision@100/500/1000 ở top các cặp cần điều tra;
- false positive theo ngôn ngữ, category, độ dài và rating;
- calibration và confusion matrix ở các threshold chính;
- community-level metrics nếu synthetic graph có nhãn cụm.

Với dữ liệu thật, nên báo cáo thống kê mô tả, stability theo seed/threshold và một protocol review thủ công nhỏ có tiêu chí rõ ràng, thay vì gọi đó là accuracy có ground truth.

## 9. Lộ trình triển khai

### Giai đoạn 1 — Proof of concept

- tải một subset có kiểm soát;
- chuẩn hóa và EDA;
- sinh synthetic pairs;
- so sánh hai baseline và một multilingual embedding;
- tạo ANN index;
- đo candidate recall và Precision@k.

### Giai đoạn 2 — Verification và ablation

- xây feature table;
- huấn luyện classifier;
- kiểm tra leakage;
- chạy ablation;
- lưu model, threshold và experiment metadata.

### Giai đoạn 3 — Dữ liệu thật và graph

- mở rộng theo category hoặc ngôn ngữ;
- chạy candidate retrieval và verification;
- xây graph với threshold đã chọn;
- tìm community và chọn case study;
- ghi chú mọi kết luận ở mức “đáng điều tra” hoặc “bất thường”.

### Giai đoạn 4 — Sản phẩm demo

- FastAPI cho scoring, tìm cặp tương tự và lấy graph neighborhood;
- dashboard hiển thị review pair, score, feature contribution, timeline và community;
- SHAP hoặc permutation importance cho classifier tabular;
- Docker hóa pipeline;
- thêm cache/index persistence để API không encode lại toàn bộ dữ liệu mỗi request.

Đối với portfolio, Streamlit có thể là lựa chọn nhanh cho dashboard. Nếu muốn thể hiện rõ năng lực full-stack, dùng FastAPI làm backend và một frontend riêng là hợp lý.

## 10. Kiểm soát chi phí tính toán

- bắt đầu với một hoặc vài category có nhiều dữ liệu;
- encode batch và lưu embedding ra disk;
- dùng float16 hoặc quantization khi phù hợp;
- xây index theo partition thay vì một index khổng lồ ngay từ đầu;
- đo candidate recall trên subset trước khi scale;
- chạy graph trên top candidate edges, không chạy all-pairs;
- tách offline batch processing khỏi online API;
- lưu intermediate artifacts để có thể chạy lại từng bước.

Một bản demo tốt không cần xử lý toàn bộ hàng triệu review. Quan trọng hơn là pipeline có metric, có kiểm soát leakage và giải thích được kết quả.

## 11. Rủi ro và cách giảm thiểu

| Rủi ro | Cách giảm thiểu |
|---|---|
| Synthetic quá dễ so với dữ liệu thật | thêm hard negatives, nhiều mức paraphrase và holdout theo transformation |
| Leakage giữa train/test | split theo source review, product, template và thời gian |
| False positive do review phổ biến | kết hợp product/time/rating, xem độ đặc hiệu của cụm và thêm review phổ biến làm negative |
| Embedding bias theo ngôn ngữ | báo cáo metric từng cặp ngôn ngữ và từng độ dài review |
| ANN bỏ sót candidate | đo candidate recall khi thay đổi k, threshold và blocking |
| Graph quá dày | giữ top-k edge hoặc threshold theo calibration |
| Overclaim về review farm | dùng ngôn ngữ “semantic duplication” và “đáng điều tra”, không gán actor |
| Chi phí encode lớn | subset theo category, batch embedding, cache và index partition |
| Dữ liệu/giấy phép không phù hợp | kiểm tra license, điều kiện sử dụng và schema trước khi công bố |
| Dashboard chậm | precompute score/graph summary, dùng API cho truy vấn nhỏ |

## 12. Tiêu chí hoàn thành tối thiểu

Đề tài được xem là hoàn thành ở mức portfolio khi có:

- một dataset card và data dictionary;
- synthetic benchmark có các mức biến đổi và hard negatives;
- baseline, multilingual embedding và ANN retrieval;
- verification classifier với PR-AUC/Precision@k được báo cáo;
- ablation trực tiếp so với translate-then-compare;
- ít nhất một graph case study trên dữ liệu thật;
- dashboard xem được cặp review, score, timeline và community;
- README giải thích rõ synthetic evaluation và real-data observation;
- Docker hoặc hướng dẫn chạy lại;
- phần limitations nêu rõ không có ground truth đầy đủ cho fake review thật.

## 13. Kết luận

Đây là một đề tài mạnh cho full-stack data science vì nó nối được nghiên cứu NLP đa ngôn ngữ, semantic search, supervised verification, graph mining và sản phẩm phân tích. Phiên bản đáng tin cậy nhất nên được định vị là:

> Hệ thống xếp hạng các cụm review có nội dung trùng lặp ngữ nghĩa xuyên ngôn ngữ và tín hiệu bất thường để hỗ trợ điều tra.

Định vị này vừa đủ tham vọng để tạo một project nổi bật, vừa trung thực với giới hạn của dữ liệu và nhãn hiện có.
