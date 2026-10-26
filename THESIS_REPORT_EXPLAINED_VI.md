# Báo cáo Luận văn (bản giải thích đầy đủ)

**Đề tài:** Học Liên kết Cá nhân hóa cho Lọc Cộng tác Bảo mật trong Hệ thống Gợi ý
*(Personalized Federated Learning for Privacy-Aware Collaborative Filtering in Recommender Systems)*
**Tác giả:** Đăng Vĩnh · **Tập dữ liệu:** MovieLens-1M · **Công cụ:** Flower (flwr) + PyTorch · **Ngày:** 2026-07-27 *(bản trước: 2026-06-15)*

> **Cách đọc báo cáo này.** Đây là bản viết lại **theo hướng giải thích** từ báo cáo tiến độ gốc (`THESIS_PROGRESS_REPORT.md`). Mọi khái niệm chuyên ngành đều được diễn giải từ đầu, dành cho người đọc **có nền tảng khoa học máy tính / học máy (machine learning) nhưng chưa từng làm về hệ thống gợi ý hay học liên kết**. Mỗi thuật ngữ tiếng Việt đều kèm thuật ngữ tiếng Anh trong ngoặc để bạn có thể tra cứu tài liệu gốc. Tên biến code, ký hiệu độ đo, tên file và con số được giữ nguyên tiếng Anh. Cuối báo cáo có **Bảng thuật ngữ** (mục 12).

---

## 0. Tóm tắt một đoạn (cho người vội)

Luận văn này định đề xuất một phương pháp gợi ý phim **vừa bảo vệ quyền riêng tư của người dùng, vừa cá nhân hóa tốt hơn** các phương pháp hiện có — đặc biệt cho những người dùng có rất ít dữ liệu. Sau khi xây dựng hệ thống đánh giá đúng chuẩn (và sửa một lỗi đo lường nghiêm trọng), kết quả thực nghiệm cho thấy phương pháp đề xuất **KHÔNG đạt mục tiêu**: nó thua hai phương pháp đơn giản hơn, và tệ nhất đúng ở nhóm người dùng ít dữ liệu — chính là nhóm mà nó được thiết kế để phục vụ. Tuy nhiên, quá trình làm việc cho ra nhiều **đóng góp có giá trị khác**: (1) phát hiện + sửa một lỗi đo lường khiến mọi kết quả trước đó bị sai lệch ~5 lần; (2) một **giải thích về cơ chế** vì sao việc "giữ dữ liệu người dùng tại máy" lại giúp mô hình tốt hơn — nay đã được củng cố bằng một phép so sánh sạch: **+20.9%** khi chỉ thay đổi đúng một yếu tố; (3) **tháng 7/2026** bổ sung ba thí nghiệm mới (mục 8): một thí nghiệm loại bỏ yếu tố gây nhiễu trong phép so sánh trung tâm, một thí nghiệm đưa PFedRec lên **0.3447 — vượt cả phiên bản huấn luyện tập trung (centralized)**, và một thí nghiệm "đo trước khi xây" đã **bác bỏ ý tưởng nâng cấp tiếp theo với chi phí gần bằng 0** nhưng đồng thời chứng minh **đầu tính điểm riêng từng người đáng giá +9.4 điểm** so với dùng một đầu chung cho cả cộng đồng. Báo cáo đề xuất chốt hướng luận văn: **kết quả phủ định trung thực + đóng góp phương pháp + chuỗi bằng chứng "cá nhân hóa nên đặt ở đâu"**.

---

## 1. Bối cảnh: ba lĩnh vực mà đề tài đứng trên

### 1.1. Hệ thống Gợi ý và Lọc Cộng tác

**Hệ thống Gợi ý (Recommender System)** là hệ thống dự đoán những món đồ (ở đây là *phim*) mà một người dùng có khả năng thích, để xếp hạng và đề xuất. Kỹ thuật phổ biến nhất là **Lọc Cộng tác (Collaborative Filtering – CF)**: ý tưởng cốt lõi là "những người có sở thích giống nhau trong quá khứ sẽ tiếp tục thích những thứ giống nhau". Ta **không** cần biết nội dung phim (thể loại, diễn viên…); chỉ cần biết *ai đã tương tác với phim nào*.

Dữ liệu được mô tả bằng một **ma trận người dùng × vật phẩm (user–item matrix)**: hàng là người dùng, cột là phim, mỗi ô cho biết người đó có xem/đánh giá phim đó hay không. Ma trận này **cực kỳ thưa (sparse)** — mỗi người chỉ xem vài chục trên hàng nghìn phim.

Cách học phổ biến là **Phân rã Ma trận (Matrix Factorization – MF)**: gán cho mỗi người dùng một **vector đặc trưng (user embedding)** và mỗi phim một **vector đặc trưng (item embedding)**, mỗi vector gồm vài chục con số ẩn (gọi là *latent factors* — nhân tố ẩn, ví dụ "mức độ hành động", "mức độ lãng mạn"…, do mô hình tự học chứ không do người gán). Mức độ một người thích một phim được ước lượng bằng **tích vô hướng (dot product)** của hai vector. Mô hình học các vector này sao cho dự đoán khớp với lịch sử tương tác.

Hai chi tiết quan trọng trong đề tài:

- **Phản hồi ngầm (implicit feedback):** ta xem dữ liệu là 0/1 ("đã tương tác / chưa") thay vì điểm số 1–5 sao. Đây là tình huống thực tế phổ biến (lượt xem, lượt click) và là chuẩn của dòng bài *Neural Collaborative Filtering (NCF)* mà luận văn so chuẩn theo.
- **Hàm mất mát xếp hạng (ranking loss):** vì mục tiêu cuối là *xếp hạng đúng thứ tự* (đưa phim đúng lên top), một số mô hình dùng **BPR (Bayesian Personalized Ranking)** — tối ưu để "phim người dùng thực sự thích phải được chấm điểm cao hơn một phim ngẫu nhiên họ chưa xem" — thay vì **MSE (Mean Squared Error)** vốn tối ưu để đoán đúng *giá trị điểm*. Đây là lý do về sau ta thấy chỉ số RMSE cao bất thường mà vẫn bình thường: mô hình BPR không cố đoán đúng điểm, nó cố xếp đúng thứ tự.

### 1.2. Học Liên kết và động cơ bảo mật

Cách huấn luyện truyền thống là gom toàn bộ dữ liệu người dùng về một máy chủ trung tâm. Điều này **vi phạm quyền riêng tư**: lịch sử xem phim là dữ liệu nhạy cảm.

**Học Liên kết (Federated Learning – FL)** giải quyết bằng nguyên tắc: **dữ liệu thô không bao giờ rời khỏi thiết bị của người dùng.** Thay vì gửi dữ liệu lên máy chủ, mỗi thiết bị (**máy khách – client**) tự huấn luyện mô hình trên dữ liệu cục bộ của nó, rồi chỉ gửi lên máy chủ **các tham số mô hình đã cập nhật** (các con số trọng số), chứ không gửi dữ liệu. Máy chủ **tổng hợp (aggregate)** các cập nhật từ nhiều máy khách thành một mô hình toàn cục, rồi gửi lại xuống. Quá trình này lặp lại theo từng **vòng (round)**.

Thuật toán tổng hợp kinh điển là **FedAvg (Federated Averaging)**: máy chủ lấy **trung bình có trọng số** các tham số nhận về từ các máy khách. Một biến thể là **FedProx**, thêm một số hạng phạt để mô hình cục bộ không trôi quá xa mô hình toàn cục (giúp ổn định khi dữ liệu các máy khách rất khác nhau).

Một điểm mấu chốt: trong FL, dữ liệu giữa các máy khách thường **không đồng nhất (non-IID)** — mỗi người có gu xem phim riêng. Điều này khiến việc lấy trung bình tham số có thể gây hại (xem mục 7).

Có hai "kịch bản" FL, và việc chọn đúng kịch bản là sống còn cho tính đúng đắn của luận văn:

- **Cross-silo (liên-tổ-chức):** vài máy khách lớn (ví dụ 5 bệnh viện), mỗi máy khách giữ dữ liệu của *nhiều* người. Cấu hình cũ của dự án (`num-supernodes=5`) thuộc loại này.
- **Cross-device (liên-thiết-bị):** rất nhiều máy khách nhỏ, **mỗi người dùng = một máy khách** (ví dụ điện thoại). Đây mới là bối cảnh mà một tuyên bố về *cá nhân hóa cho từng người* có ý nghĩa. Luận văn đã **chuyển hẳn sang kịch bản này**: N = 6040 người dùng = 6040 máy khách (xem mục 2 và 5).

### 1.3. Căng thẳng cốt lõi: toàn cục chung vs cá nhân hóa

FL muốn một mô hình **chung tốt cho tất cả**, nhưng gợi ý lại cần **cá nhân hóa cho từng người**. Đây là mâu thuẫn trung tâm. Các phương pháp khác nhau ở chỗ: **phần nào của mô hình được chia sẻ toàn cục, phần nào được giữ riêng (local) tại từng người dùng.** Toàn bộ luận văn xoay quanh việc đặt "ranh giới" này ở đâu cho tối ưu — đặc biệt cho **người dùng ít dữ liệu (sparse users)**, nhóm khó cá nhân hóa nhất vì họ gần như không có lịch sử để học.

---

## 2. Câu hỏi nghiên cứu & Tuyên bố ban đầu của luận văn

**Tuyên bố (claim) ban đầu:** Dưới kịch bản cross-device (mỗi người = một máy khách, N = 6040, tập MovieLens-1M), phương pháp đề xuất — gọi là **Adaptive / Hierarchical-Conditional** (cá nhân hóa thích nghi, sẽ giải thích ở mục 3) — phải **thắng cả ba phương pháp đối chứng** trên độ đo chất lượng xếp hạng `NDCG@10`, **đặc biệt là trên nhóm người dùng ít dữ liệu (sparse users)**. Đồng thời, phương pháp đối chứng **PFedRec** phải **tái lập (reproduce)** được kết quả đã công bố trong bài báo gốc của nó (để chứng minh hệ thống đo của ta đáng tin).

Nói cách khác, "thành công" của luận văn = mũi tên cá nhân hóa thích nghi phải là **vô địch**, nhất là ở phân khúc khó nhất. Đây là một tuyên bố **có thể kiểm chứng và có thể bị bác bỏ (falsifiable)** — và đó chính là điều xảy ra.

---

## 3. Bốn phương pháp được so sánh

Bốn phương pháp tạo thành một "bậc thang" độ phức tạp tăng dần. Sự khác biệt chính là **ranh giới giữa tham số toàn cục (chia sẻ) và tham số cục bộ (giữ riêng tại máy người dùng)**.

| # | Phương pháp | Phần TOÀN CỤC (được tổng hợp) | Phần CỤC BỘ (giữ riêng mỗi người) | Vai trò trong luận văn |
|---|---|---|---|---|
| 1 | **Baseline** (mô hình nền) | *Tất cả* tham số (kể cả vector người dùng) | (không có gì riêng) | Giới hạn dưới — phương pháp khác phải thắng nó |
| 2 | **PFedRec** | Vector phim (item embeddings) | Một "đầu tính điểm" tuyến tính riêng từng người | Phương pháp hiệu chuẩn — phải tái lập được bài báo gốc |
| 3 | **Personalized** (split-learning) | Vector phim | **Vector người dùng** (user embeddings) | Bước "giữ dữ liệu người dùng tại máy" |
| 4 | **Adaptive** | Vector phim + một "vector nguyên mẫu" toàn cục | Vector người dùng + mạng nơ-ron riêng + hệ số trộn α | **Đóng góp chính của luận văn** |

### 3.1. Baseline — "tất cả toàn cục" (giới hạn dưới)

Mọi tham số, kể cả vector của từng người dùng, đều được lấy trung bình (FedAvg) qua tất cả máy khách mỗi vòng. Tức là **không hề cá nhân hóa** — vector của bạn bị trộn lẫn với mọi người khác. Đây là mức "sàn": bất kỳ phương pháp cá nhân hóa nào cũng phải vượt qua nó, nếu không thì cá nhân hóa vô nghĩa. (Run quyết định của mô hình nền chạy `model-type=basic`, tức dùng hàm mất mát **MSE** — xem mục 1.1 — nên nó khác họ về *mục tiêu tối ưu*, một chi tiết quan trọng khi so sánh ở mục 4. Đã xác nhận trực tiếp trong `results.json` của run này; lưu ý mặc định của module lại là BPR-MF, nên MSE là lựa chọn riêng của run này.)

### 3.2. PFedRec — đầu tính điểm riêng từng người (IJCAI-23)

PFedRec (từ một bài báo tại hội nghị IJCAI-23) **không** lưu vector người dùng. Thay vào đó: vector phim được chia sẻ toàn cục, còn **mỗi người dùng có một "đầu tính điểm" (score function) tuyến tính riêng**. Cụ thể đây là một lớp `Linear(latent_dim → 1)` rồi `Sigmoid`: nó nhận vào vector của một phim (dùng chung toàn cục) và xuất ra một xác suất phù hợp *riêng theo khẩu vị người đó*; huấn luyện bằng loss **BCE** trên phản hồi nhị phân 0/1. Ba cách gọi — "đầu tính điểm", "score function", "lớp Linear", và tên biến code `affine_output` — đều chỉ **cùng một đối tượng**, và **chính lớp này *là* toàn bộ phần cá nhân hóa**. Nó được lưu trong bộ nhớ đệm cục bộ của từng người, không bao giờ gửi lên máy chủ.

### 3.3. Personalized (split-learning) — vector người dùng cục bộ

Đây là bước trung gian và là ý tưởng "split learning" (học chia tách): **vector phim chia sẻ toàn cục, nhưng vector người dùng được giữ RIÊNG tại máy mỗi người** và không bao giờ truyền đi. Trực giác: vector người dùng là thứ riêng tư và cá nhân nhất; giữ nó tại chỗ vừa bảo mật hơn vừa cho phép cá nhân hóa. (Về sau ta phát hiện lợi ích thật sự đến từ một cơ chế *khác* với trực giác này — xem mục 7.3.)

### 3.4. Adaptive — cá nhân hóa thích nghi (đóng góp chính)

Phương pháp đề xuất xây trên split-learning và thêm ba cơ chế:

1. **Hệ số trộn thích nghi α (adaptive alpha).** Với mỗi người dùng, mô hình trộn giữa "vector cục bộ của riêng họ" và "một vector nguyên mẫu toàn cục" theo công thức `p = α·(cục bộ) + (1−α)·(toàn cục)`, với α được giới hạn trong khoảng [0.1, 0.95]. Giá trị **α được tính riêng cho từng người** dựa trên đặc điểm dữ liệu của họ (số lượng tương tác, độ đa dạng thể loại, độ nhất quán đánh giá…). Ý tưởng: **người ít dữ liệu (sparse) nên dựa nhiều vào phần toàn cục (α thấp), người nhiều dữ liệu nên tự tin dùng phần cục bộ (α cao)** — vì người ít dữ liệu không đủ lịch sử để học một vector cá nhân đáng tin, nên dựa vào "trí tuệ đám đông" (phần toàn cục) là an toàn hơn. Đây chính là cơ chế kỳ vọng sẽ "cứu" nhóm sparse. *(Lưu ý: trong run quyết định ở mục 6, α là giá trị heuristic tính theo đặc điểm người dùng; cơ chế "học α theo gradient cho từng người" — `enable-per-user-alpha` — đã được tắt, xem mục 10.)*
2. **Cá nhân hóa hai tầng (dual-level).** Ngoài việc trộn vector (tầng thống kê), còn có một **mạng nơ-ron nhỏ riêng từng người (PersonalMLP)** để tính điểm (tầng phi tuyến).
3. **Vector nguyên mẫu toàn cục (global prototype).** Máy chủ duy trì một vector "đại diện trung bình của cả cộng đồng", cập nhật dần theo **trung bình trượt có trọng số mũ (Exponential Moving Average – EMA): mỗi vòng giữ lại 90% giá trị cũ và trộn vào 10% giá trị mới** (`p_global = 0.9·cũ + 0.1·mới`), rồi gửi xuống mỗi vòng. Đây là kênh để người ít dữ liệu "mượn" tri thức từ đám đông.

Tóm lại, Adaptive là phương pháp **phức tạp nhất, nhiều tham số riêng nhất**, và được thiết kế nhắm thẳng vào việc giúp nhóm sparse.

---

## 4. Phương pháp đánh giá (cách chúng tôi đo "tốt")

Để so sánh công bằng, cả bốn phương pháp dùng **chung một quy trình đánh giá (protocol)** đã được "khóa" (cố định). Dưới đây là cách đo và ý nghĩa từng yếu tố.

**Cách kiểm tra — "bỏ-một-ra + 99 mẫu âm" (leave-one-out + 99 negatives).** Với mỗi người dùng: lấy *phim cuối cùng họ tương tác* làm đáp án đúng cần đoán (giấu đi), trộn nó với **99 phim họ chưa từng xem** (gọi là *mẫu âm – negatives*), rồi yêu cầu mô hình **xếp hạng 100 phim này**. Mô hình tốt sẽ đẩy phim-đáp-án lên gần đỉnh. Đây là chuẩn đánh giá của dòng NCF, giúp số liệu so sánh được với các bài báo khác.

**Các độ đo (metrics) — đều "càng cao càng tốt", giá trị 0→1:**
- **`NDCG@10` (Normalized Discounted Cumulative Gain tại top-10):** độ đo *chất lượng xếp hạng*. Không chỉ hỏi "phim đúng có lọt top-10 không" mà còn thưởng điểm cao hơn nếu nó nằm **càng gần vị trí số 1**. Vì ở giao thức này **mỗi người chỉ có đúng 1 phim-đáp-án**, NDCG@10 rút gọn thành `1/log2(vị_trí + 1)` nếu phim đúng lọt top-10 (và 0 nếu không) — tức một dạng `HR@10` **có trọng số theo vị trí** (rất gần với MRR). Đây là **độ đo chính** của luận văn.
- **`HR@10` (Hit Rate tại top-10):** đơn giản hơn — *tỉ lệ* trường hợp phim đúng lọt vào top-10 (bất kể vị trí nào trong top-10).
- **`MRR` (Mean Reciprocal Rank):** trung bình của các *nghịch-đảo-vị-trí* (lấy `1/vị_trí` cho từng người rồi mới trung bình — **không phải** nghịch đảo của vị-trí-trung-bình). Độ đo này được dòng NCF dùng nhưng không xuất hiện trong kết quả cuối của chúng tôi nên báo cáo không dùng.

**Phân nhóm người dùng theo mật độ dữ liệu** — vì điều luận văn quan tâm nhất là nhóm ít dữ liệu:
- **sparse (ít dữ liệu):** 0–30 lượt tương tác — *nhóm cốt lõi của tuyên bố luận văn.*
- **medium (trung bình):** 30–100 lượt.
- **dense (dày):** trên 100 lượt.

**Bảng quy trình đã khóa** (cả 4 run dùng y hệt):

| Thông số (knob) | Giá trị | Giải thích |
|---|---|---|
| Partition (cách chia dữ liệu) | `natural` — cross-device, **mỗi người = 1 máy khách, N=6040** | Kịch bản đúng cho tuyên bố cá nhân hóa |
| Cách kiểm tra | leave-one-out + 99 negatives | Chuẩn NCF |
| Độ đo chính | `sampled_ndcg@10` (kèm `HR@10`) | "sampled" = đo trên 100 phim đã lấy mẫu |
| Nhóm người dùng | sparse / medium / dense | Theo mục trên |
| Tỉ lệ chọn máy khách mỗi vòng | `fraction_train = 0.1` (10%, ký hiệu **C=0.1**) | Mỗi vòng chỉ ~10% người tham gia (giống thực tế) |
| Số vòng | `num_server_rounds = 100` (cả 4 chạy đủ 100) | |
| Cách chọn mô hình cuối | `best_round_restore` — khôi phục về **vòng có `NDCG@10` cao nhất** | Tránh lấy nhầm vòng cuối nếu nó kém hơn |
| Phạm vi đánh giá cuối | **toàn bộ 6040 người** (`evaluated_users = 6040`) | Đánh giá trên *cả dân số*, không chỉ mẫu nhỏ |
| Seed (hạt giống ngẫu nhiên) | 42 (hiện mới chỉ 1 seed) | Xem giới hạn ở mục 10 |

Mọi con số ở mục 5 được **chép nguyên văn** từ file kết quả `final_metrics.best` của từng run, không làm tròn.

---

## 5. Một lỗi đo lường đã được phát hiện và sửa (đóng góp phương pháp #1)

Trước khi tin bất kỳ con số nào, chúng tôi phát hiện **một lỗi nghiêm trọng ở khâu đo lường** (không phải ở mô hình) đã làm sai lệch mọi kết quả trước đó. Đây hóa ra là một đóng góp có giá trị riêng.

**Lỗi "đánh giá nguội" (cold-evaluation bug).** Ở lượt đánh giá cuối cùng trên toàn bộ dân số (gọi nội bộ là *D-06*), hệ thống lẽ ra phải nạp **tham số cục bộ đã-được-huấn-luyện của từng người** (đầu tính điểm riêng, vector người dùng riêng…). Nhưng do một thiếu sót kỹ thuật — lượt đánh giá không gắn đúng định danh `run_id` vào cấu hình — hệ thống lại đọc nhầm từ một **thư mục bộ nhớ đệm mặc định (`.embedding_cache/default/`) chứa tham số cục bộ CHƯA huấn luyện** (vừa khởi tạo ngẫu nhiên), thay vì thư mục riêng đã được huấn luyện của từng người. Hậu quả: **mọi người dùng bị chấm điểm bằng một mô hình cá nhân "nguội" (cold).** Kết quả gần như ngẫu nhiên.

Lỗi này **chỉ ảnh hưởng phần tham số CỤC BỘ** — vì chỉ phần cục bộ mới phải nạp lại từ bộ nhớ đệm theo từng người lúc đánh giá; phần toàn cục thì luôn được máy chủ gửi mới (đã huấn luyện) nên không hề bị "nguội". Do đó **mức thiệt hại tỉ lệ thuận với việc *bao nhiêu phần của "bộ tính điểm" nằm ở phần cục bộ*.** Đo trực tiếp trên PFedRec (nơi *toàn bộ* bộ tính điểm là cục bộ): chỉ số rơi từ **0.3478 (đúng, "ấm/warm")** xuống **0.0711 (nguội/cold)** — sai khoảng **5 lần**. Lỗi đã được sửa riêng cho từng module:

- `01d8b72` — PFedRec
- `3a393fb` — Personalized
- `70a92bd` — Adaptive

Riêng **Baseline miễn nhiễm** với lỗi này: vì nó dựng lại *toàn bộ* mô hình từ các tham số toàn cục được gửi xuống, nó không có "phần cục bộ" nào để đọc nhầm.

**Vì sao chi tiết này quan trọng:** nó giải thích vì sao *các kết quả trước đây không đáng tin* và vì sao toàn bộ thí nghiệm phải chạy lại sau khi sửa. Bài học "phải kiểm chứng warm-vs-cold trước khi tin số" là một đóng góp chuyển giao được cho cộng đồng đánh giá hệ gợi ý liên kết.

---

## 6. Kết quả chính

Bảng dưới là kết quả của **run quyết định** cho mỗi phương pháp (tất cả: N=6040, đánh giá "ấm" trên toàn dân số, lấy vòng tốt nhất). **NDCG@10 và HR@10 — càng cao càng tốt.**

| Phương pháp | Run ID | Vòng tốt nhất | NDCG@10 (tổng) | NDCG@10 *sparse* | NDCG@10 medium | NDCG@10 dense | HR@10 (tổng) | HR@10 sparse |
|---|---|---|---|---|---|---|---|---|
| **PFedRec + tinh chỉnh cuối** *(mới, 07/2026 — xem mục 8.2)* | `…cf72e0` | 83 | **0.34471** | **0.42697** | **0.38016** | **0.29354** | **0.61109** | **0.70581** |
| **PFedRec** | `…ef41ab` | 100 | 0.33518 | 0.40567 | 0.37104 | 0.28697 | 0.59238 | 0.67120 |
| **Personalized** | `…f18e64` | 33 | 0.22748 | 0.27263 | 0.24509 | 0.20087 | 0.41970 | 0.49197 |
| **Adaptive** *(đề xuất)* | `…869730` | 83 | 0.21436 | 0.23546 | 0.22641 | 0.19887 | 0.38675 | 0.40915 |
| **Baseline** (MSE, dim 128) | `…1bf513` | 25 | 0.20131 | 0.24458 | 0.22171 | 0.17300 | 0.36457 | 0.42645 |
| **Baseline BPR** *(mới, 07/2026 — cùng cấu hình với Personalized, xem mục 8.1)* | `…e81518` | 83 | 0.18817 | 0.24687 | 0.19935 | 0.16292 | 0.32583 | 0.40049 |

**Xếp hạng theo NDCG@10 tổng:** PFedRec+tinh chỉnh (0.34471) > PFedRec (0.33518) > Personalized (0.22748) > **Adaptive (0.21436)** > Baseline MSE (0.20131) > Baseline BPR (0.18817).
**Xếp hạng theo NDCG@10 nhóm sparse:** PFedRec+tinh chỉnh (0.42697) > PFedRec (0.40567) > Personalized (0.27263) > Baseline BPR (0.24687) > Baseline MSE (0.24458) > **Adaptive (0.23546)** — Adaptive vẫn bét, kể cả so với baseline yếu hơn mới thêm.

**Mốc so sánh ngoài-federated:** cùng quy trình đánh giá này, phiên bản **huấn luyện tập trung (centralized — gom hết dữ liệu về một chỗ, không bảo vệ riêng tư)** của BPR-MF đạt NDCG@10 = **0.3332**. Tức là PFedRec federated **đạt/vượt centralized** — một câu bảo vệ rất mạnh cho hướng "riêng tư mà không mất chất lượng".

> ⚠️ **Lưu ý đây là so sánh "cấp hệ thống", không phải so từng-nút-một (knob-for-knob — tức chỉ thay đổi đúng MỘT thiết lập và giữ mọi thứ khác y hệt).** Bốn phương pháp khác nhau ở nhiều trục thiết kế, nên cần đọc kết quả thận trọng:
> - **Mục tiêu tối ưu khác nhau:** Baseline dùng **MSE** (đoán điểm), còn Personalized/Adaptive dùng **BPR** (xếp hạng).
> - **Kích thước vector khác nhau:** Baseline=128, Personalized=64, Adaptive=64; PFedRec dùng `latent_dim=32` nguyên bản của nó.
> - **PFedRec có "công thức huấn luyện" riêng** (tốc độ học lớn cho vector phim…), không so trực tiếp từng-nút với họ MF.
> - **Thuật toán tổng hợp (FL strategy) khác nhau:** Adaptive là module duy nhất trong nhóm đóng góp dùng `fedprox`; Baseline/Personalized có lưu `proximal_mu=0.01` nhưng thực chất chạy `FedAvg`, nên hệ số μ bị vô hiệu (inert).
>
> Do đó ưu thế của PFedRec nên đọc là *"một thiết kế kiểu đầu-riêng-từng-người thắng"*, chứ không phải "thua knob-for-knob".

---

## 7. Diễn giải: ba kết luận

Run quyết định của phương pháp đề xuất là Adaptive **`…869730`** (hoàn tất 2026-06-14), với các "cổng tin cậy" (trust gates) đều đạt (xem mục 9). Ba kết luận:

### 7.1. Tuyên bố ban đầu của luận văn BỊ BÁC BỎ (falsified)

Phương pháp đề xuất lẽ ra phải thắng cả ba đối chứng trên NDCG@10, *đặc biệt ở nhóm sparse*. Thực tế, Adaptive:
- chỉ thắng **Baseline** về tổng (+6.5%);
- **thua** Personalized (−5.8%) và **thua đậm** PFedRec (−36%);
- và **xếp BÉT ở nhóm sparse (0.23546)** — thấp hơn cả Baseline (0.24458) — tức **ngược hẳn** điều nó được thiết kế để làm.

Cơ chế trộn α và cá nhân hóa hai tầng, dưới cấu hình này, **không tạo ra sự "chuyển giao tri thức" cho nhóm sparse** như kỳ vọng.
*(Lưu ý thống kê: các phần trăm so sánh này dựa trên 1 seed; khoảng cách giữa Adaptive/Personalized/Baseline là nhỏ và chưa được chứng minh là tách biệt về mặt thống kê — xem mục 10. Riêng khoảng cách lớn của PFedRec và việc "Adaptive bét ở sparse" thì đủ vững.)*

### 7.2. PFedRec tái lập đúng (trong giới hạn tài nguyên khả thi)

PFedRec đạt **NDCG@10 ≈ 0.335 / HR@10 ≈ 0.59**, khớp mốc kỳ vọng cho kịch bản này. Mốc cao hơn trong bài báo gốc (HR@10 0.65–0.70, NDCG@10 0.35–0.42) đòi hỏi *toàn bộ người dùng tham gia mỗi vòng* (`fraction=1.0`, tốn ~6 ngày tính toán) — một điều **bất khả thi về tài nguyên đã được ghi nhận, không phải tái lập thất bại**. Bằng chứng khẳng định: con số cross-device 0.335 nằm **ngay dưới** band full-participation của bài báo (0.35–0.42), **đúng như giới hạn tài nguyên (compute envelope) dự đoán** — tức đây là một lần tái lập *thành công* trong phạm vi khả thi. Việc PFedRec mạnh nhất ở đây tự nó là một thông tin: **thiết kế "đầu tính điểm riêng từng người trên vector phim chung" lại thắng cả split-learning lẫn phương pháp thích nghi phức tạp.**

### 7.3. Điểm sáng có thể công bố: cơ chế của split-learning (đóng góp #2)

Personalized (split-learning) thắng Baseline **+13% về tổng** (0.22748 vs 0.20131) và ở **mọi** nhóm mật độ. **Giả thuyết khả dĩ nhất** về *vì sao* lại **không phải** "cá nhân hóa nhờ vector người dùng riêng", mà là:

> Bằng cách **giữ vector người dùng tại máy** (không gửi đi để lấy trung bình), ta **bảo vệ quá trình huấn luyện vector phim toàn cục khỏi bị "nhiễu" do trung bình hóa các vector người dùng rất khác nhau (non-IID).** Nhờ đó bảng vector phim chung hội tụ về một biểu diễn *sạch hơn*, tốt cho tất cả.

*Vì sao trung bình hóa vector người dùng lại làm hại bảng vector PHIM?* Vector người dùng và vector phim được học **chung** qua cùng một phép tích vô hướng: mỗi vòng, mô hình điều chỉnh vector phim *dựa trên* vector người dùng hiện tại. Trong Baseline (mọi thứ toàn cục), máy chủ lấy trung bình các vector người dùng cực kỳ khác nhau (mỗi người một gu) thành một "người dùng trung bình" méo mó, vô nghĩa; gradient cập nhật vector phim bị tính theo cái vector méo đó nên bảng phim bị nhiễu. Khi giữ vector người dùng tại máy (split-learning / PFedRec), mỗi máy khách cập nhật vector phim theo *đúng* vector người dùng thật của nó, rồi chỉ trung bình hóa **các cập nhật vector phim** — sạch hơn nhiều. Đó là lý do giả thuyết cho rằng lợi ích đến từ "bảo vệ huấn luyện vector phim", không phải từ bản thân việc cá nhân hóa.

Giả thuyết này **nhất quán với** (chứ chưa *chứng minh*) việc PFedRec — vốn cũng loại vector người dùng khỏi quá trình tổng hợp — thắng tuyệt đối; lưu ý ưu thế của PFedRec cũng có thể một phần đến từ sức biểu đạt của đầu-riêng-từng-người, chứ không chỉ từ việc bảo vệ bảng phim. Tuy nhiên, đây **mới là giả thuyết, CHƯA được kiểm chứng bằng thực nghiệm** — cần **một thí nghiệm đối chứng (ablation)** để xác nhận (ví dụ: so "vector người dùng cục bộ" với "vector người dùng cố định/toàn cục", giữ nguyên phần phim). Phải trình bày như **giả thuyết, không phải kết quả**, cho đến khi chạy ablation đó.

---

## 8. Cập nhật tháng 7/2026 — ba thí nghiệm mới

Sau bản báo cáo tháng 6, ba thí nghiệm được thực hiện theo đúng thứ tự ưu tiên đã được một vòng **phản biện nội bộ** (ngày 06/07, gồm nhiều bên đánh giá độc lập) chốt lại. Mỗi thí nghiệm trả lời một câu hỏi cụ thể.

### 8.1. Thí nghiệm 1 — Loại bỏ yếu tố gây nhiễu trong phép so sánh trung tâm

**Câu hỏi:** claim "+13% nhờ split-learning" ở mục 7.3 so sánh hai hệ thống khác nhau ở *nhiều* điểm cùng lúc (khác hàm mất mát MSE/BPR, khác kích thước vector 128/64). Vậy +13% đến từ split-learning, hay chỉ từ việc dùng loss BPR tốt hơn?

**Cách làm:** chạy lại Baseline với **đúng từng thiết lập** của Personalized (BPR, vector 64 chiều, cùng tốc độ học, cùng mọi thứ) — chỉ khác đúng MỘT chỗ: vector người dùng bị gửi lên máy chủ lấy trung bình (all-global) thay vì giữ riêng tại máy (split-learning). Đây chính là phép so **từng-nút-một (knob-for-knob)** mà mục 6 nói là còn thiếu.

**Kết quả:** Baseline-BPR đạt 0.18817; Personalized đạt 0.22748 → split-learning thắng **+20.9%** (nhóm sparse +10.4%, HR@10 +28.8%). Yếu tố gây nhiễu bị loại bỏ, và **theo hướng có lợi**: claim cũ (+13%) hóa ra còn *khiêm tốn* so với sự thật. Đây giờ là phép so sánh nhân-quả sạch nhất của luận văn: chỉ một thay đổi, gần +21% chất lượng.

*(Ghi chú trung thực: Baseline-BPR (0.18817) thấp hơn cả Baseline-MSE (0.20131) — gợi ý rằng việc trung bình hóa vector người dùng phá loss BPR nặng hơn — nhưng hai run này lệch kích thước vector (64 vs 128) nên không trích con số này như một kết luận thuần về loss.)*

### 8.2. Thí nghiệm 2 — "Tinh chỉnh cuối" cho PFedRec: mốc so sánh mới 0.3447

**Câu hỏi:** trong huấn luyện, mỗi vòng chỉ ~10% người dùng tham gia, nên đến lượt đánh giá cuối, "đầu tính điểm" của đa số người dùng đã **cũ (stale)** — được học từ nhiều vòng trước, không còn khớp với bảng vector phim mới nhất. Nếu cho mỗi người **một lượt học ngắn để căn lại đầu của mình** ngay trước khi đánh giá (gọi là *tinh chỉnh cuối — final calibration*; bảng phim được đóng băng), ta gỡ lại được bao nhiêu?

**Kết quả (run ~8 giờ):** PFedRec tăng từ 0.33518 → **0.34471**, tức đóng được **~45%** khoảng cách đến mức trần lý thuyết. Nhóm sparse hưởng lợi nhiều nhất (+2.1 điểm → 0.42697). Mọi cổng tin cậy đều đạt, kể cả chỉ số trực tiếp `d06_cache_misses = 0`.

**Ba hệ quả:** (1) từ nay mọi cải tiến quanh "đầu tính điểm" phải so với mốc **0.3447**, không phải 0.3352; (2) khoảng trống còn lại để cải thiện chỉ còn ~0.0115 — rất hẹp; (3) con số này **vượt cả phiên bản centralized** (0.3332) — xem mục 6.

### 8.3. Thí nghiệm 3 — Ý tưởng nâng cấp "α-PFedRec" bị bác bỏ ngay ở bước đo-trước-khi-xây

**Ý tưởng đề xuất (capstone):** mỗi người dùng gửi thêm đầu tính điểm của mình (chỉ 32 con số) lên máy chủ; máy chủ lấy trung bình thành một **"đầu dân số" (global head)**; khi chấm điểm, mỗi người dùng **pha trộn** đầu riêng với đầu dân số theo tỉ lệ α: `đầu_dùng = α·(đầu riêng) + (1−α)·(đầu dân số)`. Kỳ vọng: người ít dữ liệu (đầu riêng kém tin cậy) sẽ hưởng lợi khi mượn thêm "trí tuệ đám đông".

**Nguyên tắc "đo trước khi xây":** thay vì lập trình toàn bộ cơ chế này vào hệ thống liên kết rồi chạy ~8 giờ GPU mới biết kết quả, chúng tôi xây một **bản tái lập ngoại tuyến (offline replay)**: nạp lại đúng mô hình + đầu tính điểm của cả 6040 người từ run ở mục 8.2, rồi thử MỌI giá trị α trên máy tính thường trong vài phút. Trước khi tin bất kỳ con số nào, bản tái lập phải vượt một **cổng tin cậy**: chấm lại toàn bộ 6040 người và phải ra **đúng từng chữ số** so với kết quả gốc của run — và nó đã khớp **tuyệt đối (sai lệch = 0)**.

Sau đó, câu hỏi then chốt: *nếu có một "nhà tiên tri" (oracle — biết trước đáp án) chọn giùm mỗi nhóm người dùng giá trị α tốt nhất có thể, kết quả có vượt nổi việc cứ dùng đầu riêng 100% không?* (Có kiểm tra chéo — chọn α trên một nửa dân số, áp cho nửa kia — để "nhà tiên tri" không thắng nhờ may mắn.)

**Kết quả — KHÔNG, và rất dứt khoát:**

| Phép đo | Kết quả | Ý nghĩa |
|---|---|---|
| Oracle theo nhóm (kiểm tra chéo) | 0.3445 vs 0.3447 = **−0.0002** | ngay cả khi biết trước đáp án cũng KHÔNG thắng nổi đầu-riêng-thuần |
| α tối ưu toàn cục | **1.0** (ở cả 3 cách chia nhóm) | "pha 0% đầu dân số" chính là tối ưu |
| Công thức α heuristic (mượn từ phương pháp Adaptive) | 0.3061 = **−3.9 điểm** | công thức tự động còn gây HẠI — đúng như vòng phản biện đã cảnh báo trước |
| Nhóm sparse | −0.0015 | việc pha còn làm hại đúng nhóm cần giúp |

**Phán quyết theo ngưỡng đã đăng ký trước: DỪNG (NO-GO).** Không lập trình tiếp, không chạy GPU — tiết kiệm ~4 ngày công + ~8 giờ GPU cho một hướng không có tín hiệu.

**Nhưng phần giá trị nhất lại nằm ở các phép đo "phản chứng" (counterfactual) thu được từ cùng lượt quét** — trả lời câu hỏi *"đầu tính điểm riêng từng người thực sự đáng giá bao nhiêu?"*:

| Dùng đầu nào để chấm điểm? | NDCG@10 | Đọc thế nào |
|---|---|---|
| Đầu riêng của chính mình (α=1) | **0.34471** | chuẩn |
| **Đầu dân số (trung bình 6040 người, α=0)** | **0.25099** | mất **−9.4 điểm** → đầu riêng là thứ *gánh* chất lượng |
| Đầu của một người NGẪU NHIÊN khác | 0.19733 | đầu riêng thực sự "cá nhân", không phải ai dùng cũng được |
| Đầu khởi tạo ngẫu nhiên (chưa học) | 0.07412 | mức sàn |

Câu chuyện rút gọn cho luận văn: **có một đầu tính điểm riêng cho từng người đáng giá +9.4 điểm; còn việc *pha* đầu riêng với đầu dân số không thêm được gì. Giá trị nằm ở chỗ CÓ phần riêng, không phải ở chỗ TRỘN nó.** Một kết quả phủ định sạch (ý tưởng α-blend chết) đi kèm một kết quả khẳng định mạnh (cá nhân hóa đặt ở đầu tính điểm là đúng chỗ) — cả hai đều dùng được cho luận văn, và cả hai đều có "tem" tin cậy tuyệt đối nhờ cổng tái lập khớp-từng-chữ-số.

---

## 9. Độ tin cậy của kết quả (trust gates)

Cả bốn run được kiểm định độc lập qua các file "tem chứng nhận" `EVAL_VALIDITY.json`. Toàn dự án có **40 lượt đánh giá đã lưu tem** (gồm nhiều run thử nghiệm cũ trong suốt quá trình) — **7 hợp lệ (valid), 14 dính lỗi cold-eval (invalid_cold_eval), 19 chỉ-để-chẩn-đoán (diagnostics_only)**; trong đó **6 run chính thức ở mục 6** được dùng cho kết luận. Hai run mới (07/2026) đều mang bằng chứng "ấm" mạnh: run PFedRec+tinh chỉnh có `d06_cache_misses = 0` được ghi trực tiếp; run Baseline-BPR miễn nhiễm về cấu trúc. Từ 07/2026 còn thêm một tầng tin cậy mới: **cổng tái lập (replay trust gate)** — công cụ đánh giá ngoại tuyến phải chấm lại toàn bộ 6040 người và khớp đúng-từng-chữ-số với kết quả gốc trước khi bất kỳ phân tích nào được tin (trên run cf72e0: khớp tuyệt đối, sai lệch = 0).

**Lưu ý tinh tế — nhãn "valid" CHƯA đủ để chứng minh đánh giá đã "ấm".** Luật gán nhãn valid chỉ kiểm tra *code có chứa bản vá hay chưa*, chứ không đảm bảo một lượt đánh giá *cụ thể* đã chạy ấm. Bằng chứng **trực tiếp** duy nhất là chỉ số `d06_cache_misses = 0` (số lần đọc nhầm bộ nhớ đệm = 0). Phản ví dụ ngay trong dữ liệu của chúng tôi: một run Adaptive *thứ hai* (`…6394ca`, bị dừng sớm) cũng mang nhãn valid nhưng vẫn **rơi xuống NDCG@10 ≈ 0.080** vì không có chỉ số gate này. Vậy: **nhãn valid là điều kiện cần, chỉ số `d06_cache_misses=0` mới là điều kiện đủ.**

| Phương pháp | Run | Bằng chứng "đánh giá ấm" trực tiếp |
|---|---|---|
| **Adaptive** | `…869730` | **`d06_cache_misses = 0`** (mạnh nhất — gate trực tiếp) |
| PFedRec | `…ef41ab` | Không có chỉ số này, nhưng NDCG@10 = 0.3352 *khớp mốc ấm đã biết* (nếu nguội sẽ rơi về ~0.07) → bản vá đã có hiệu lực (gián tiếp) |
| Personalized | `…f18e64` | Không có chỉ số; nhưng code chứa bản vá `3a393fb`; nhất quán với "ấm" (gián tiếp) |
| Baseline | `…1bf513` | Miễn nhiễm về cấu trúc — không có bộ nhớ cục bộ nên không có khái niệm nguội/ấm |

**Kết luận:** run Adaptive *quyết định* mang bằng chứng mạnh nhất (chỉ số trực tiếp = 0). Ba run còn lại "ấm" theo suy luận gián tiếp; việc gắn thêm chỉ số `d06_cache_misses` cho chúng là một hạng mục kiểm tra rẻ tiền nên làm cho bảng kết quả cuối.

---

## 10. Giới hạn & các yếu tố gây nhiễu (confounds)

1. **Mới chỉ 1 seed (42).** Mọi con số đều từ một lần gieo ngẫu nhiên duy nhất. Chưa chạy **ma trận 3-seed** nên *chưa thể* gắn khoảng tin cậy hay khẳng định khác biệt thống kê giữa ba ô gần nhau (Adaptive 0.21436 vs Personalized 0.22748 vs Baseline 0.20131). Khoảng cách của PFedRec đủ lớn để vững; thứ tự giữa ba phương pháp còn lại thì **chưa chắc chắn**. *Đây là khoảng trống quan trọng nhất phải đóng trước khi nộp.*
2. **Hiệu chuẩn không đồng đều (calibration non-uniformity).** Pipeline Adaptive có thêm một bước **"tinh chỉnh cuối" (final-calibration: chạy thêm 1 epoch trên *tất cả* người dùng sau khi khôi phục vòng tốt nhất, để căn lại các đầu cá nhân hóa trước khi đánh giá)** mà các đối chứng không có. Đây là lập luận phân tích chứ không phải đại lượng đo, nhưng nếu có tác dụng thì nó chỉ **làm đẹp cho Adaptive** — nên nó *củng cố* (không làm yếu) kết luận phủ định.
3. **Cấu hình chạy không phải "mặc định thuần".** Run Adaptive đã *tắt* hai kỹ thuật tùy chọn (`enable-per-user-alpha`, `enable-item-perturbation`), **nhưng** vẫn *bật* một số hạng phụ trợ `contrastive_lambda = 0.1` (một dạng loss đối lập). Phân tích trước cho thấy số hạng này gần như vô hại trên kết quả toàn dân số, nên không đổi kết luận, song nên dán nhãn run là "dual + contrastive" cho chính xác. (Ngoài ra `per-user-alpha` còn dính một lỗi cũ chưa sửa, nên chưa phải đòn bẩy sạch.)
4. **Khác biệt kiến trúc/mục tiêu/thuật toán** (như mục 6): khác hàm mất mát, kích thước vector, công thức tối ưu. So sánh công bằng *trong họ BPR/MF*; PFedRec là kiến trúc khác nên ưu thế của nó cần đọc theo nghĩa thiết kế.

---

## 11. Đề xuất bước tiếp theo (cần ý kiến giáo sư hướng dẫn)

Câu hỏi "Path A hay Path B" của bản tháng 6 đã được trả lời phần lớn **bằng dữ liệu** trong tháng 7: phiên bản tham vọng nhất của hướng "cố cứu ý tưởng cá nhân hóa thích nghi" (capstone α-PFedRec, mục 8.3) đã được kiểm tra bằng phương pháp đo-trước-khi-xây và **bị bác bỏ với chi phí gần bằng 0**. Khuyến nghị hiện tại: chốt khung luận văn gồm **bốn khối**, tất cả đã có số liệu:

1. **Đóng góp phương pháp:** lớp lỗi cold-eval (sai ~5 lần, dính 3/4 module) + chuyển sang quy trình cross-device đúng chuẩn + hạ tầng "cổng tin cậy" (kể cả cổng tái lập khớp-từng-chữ-số) — bài học chuyển giao được cho cộng đồng.
2. **Kết quả khẳng định:** split-learning **+20.9%** trong phép so từng-nút-một (mục 8.1); PFedRec + tinh chỉnh cuối đạt **0.3447, vượt centralized 0.3332** (mục 8.2).
3. **Chuỗi bằng chứng "cá nhân hóa nên đặt ở đâu":** đầu riêng đáng +9.4 điểm so với đầu dân số; đầu của người khác tệ hơn 14.7 điểm; nhưng pha α không thêm gì (mục 8.3) — một khảo sát có số liệu ở mọi mốc.
4. **Hai kết quả phủ định trung thực, có phân tích nguyên nhân:** phương pháp Adaptive gốc (mục 7.1) và cơ chế pha α ở đầu tính điểm (mục 8.3).

**Việc còn lại trước khi viết:**
- **(Bắt buộc)** Ý nghĩa thống kê: hiện mọi số vẫn từ 1 seed. Kế hoạch rẻ nhất: dùng bảng điểm **per-user** (6040 dòng/hệ thống — đã bắt đầu sinh tự động từ 07/2026) để làm **kiểm định theo cặp** (Wilcoxon signed-rank: so hai hệ thống trên CÙNG từng người dùng); nếu tài nguyên cho phép, chạy thêm 2 seed cho các ô sát nhau.
- **(Bắt buộc, không tốn GPU)** Đọc + tóm tắt 6 bài báo liên quan còn thiếu cho chương related-work (FedRecon, FedTHE, FedRAP, APFL, FedRep, GPFedRec) — FedTHE là công trình gần ý tưởng α-blend nhất, bắt buộc phải trích khi kể chuyện mục 8.3.
- **(Bắt buộc, không tốn GPU)** Một mục "mô hình mối đe dọa" (threat model) + bảng so sánh lượng dữ liệu truyền đi của từng phương pháp — vì tên đề tài có chữ "Privacy-Aware" mà báo cáo chưa đo gì về riêng tư.
- (Tùy chọn) Một thí nghiệm đối chứng cho giả thuyết cơ chế ở mục 7.3 (đóng băng vector người dùng) nếu muốn nâng "giả thuyết" thành "kết quả".

**Ba câu hỏi cần giáo sư quyết:** (1) đồng ý chốt khung bốn khối ở trên? (2) mức đầu tư cho ý nghĩa thống kê: chỉ kiểm định theo cặp trên seed 42, hay thêm 2 seed cho các ô sát nhau? (3) nếu đổi khung thì thủ tục đổi tên đề tài cần bao lâu?

---

## 12. Bảng thuật ngữ (Việt ↔ Anh)

| Tiếng Việt (dùng trong báo cáo) | Tiếng Anh / ký hiệu | Giải thích ngắn |
|---|---|---|
| Hệ thống gợi ý | Recommender System | Hệ dự đoán & xếp hạng món người dùng có thể thích |
| Lọc cộng tác | Collaborative Filtering (CF) | Gợi ý dựa trên "người giống nhau thích thứ giống nhau" |
| Phân rã ma trận | Matrix Factorization (MF) | Biểu diễn người dùng & vật phẩm bằng vector ẩn |
| Vector đặc trưng / vector nhúng | embedding | Vector số học biểu diễn 1 người hoặc 1 phim |
| Vector người dùng / vật phẩm | user / item embedding | Vector của một người dùng / một phim |
| Phản hồi ngầm | implicit feedback | Dữ liệu 0/1 (đã/chưa tương tác) thay vì điểm sao |
| Hàm mất mát | loss function | Đại lượng mô hình tối thiểu hóa khi học |
| BPR | Bayesian Personalized Ranking | Loss tối ưu *thứ tự xếp hạng* |
| MSE | Mean Squared Error | Loss tối ưu *giá trị điểm dự đoán* |
| Học liên kết | Federated Learning (FL) | Huấn luyện mà dữ liệu thô không rời thiết bị |
| Máy khách / máy chủ | client / server | Thiết bị người dùng / máy trung tâm tổng hợp |
| Vòng | round | Một chu kỳ huấn luyện-cục-bộ + tổng-hợp |
| Tổng hợp | aggregation | Gộp cập nhật từ nhiều máy khách |
| FedAvg / FedProx | — | Thuật toán tổng hợp (trung bình / có phạt) |
| Liên-thiết-bị / liên-tổ-chức | cross-device / cross-silo | Mỗi người=1 máy khách / vài máy khách lớn |
| Không đồng nhất | non-IID | Dữ liệu các máy khách khác phân bố nhau |
| Tham số toàn cục / cục bộ | global / local params | Phần chia sẻ / phần giữ riêng từng người |
| Cá nhân hóa | personalization | Điều chỉnh mô hình riêng cho từng người |
| Hệ số trộn α | alpha (α) | Tỉ lệ trộn giữa phần cục bộ và toàn cục |
| Vector nguyên mẫu | (global) prototype | Vector "đại diện đám đông" do máy chủ giữ |
| Trung bình trượt mũ | EMA (Exponential Moving Average) | Cập nhật dần: giữ 90% cũ + 10% mới mỗi vòng |
| Người dùng thưa / dày | sparse / dense users | Người ít / nhiều lượt tương tác |
| Mẫu âm | negative samples | Phim người dùng *chưa* tương tác, dùng để đối chứng |
| Bỏ-một-ra | leave-one-out | Giấu 1 tương tác cuối làm đáp án cần đoán |
| Độ đo | metric | Đại lượng đo chất lượng (NDCG@10, HR@10…) |
| NDCG@10 | — | Chất lượng xếp hạng top-10 (thưởng vị trí cao) |
| HR@10 | Hit Rate@10 | Tỉ lệ phim đúng lọt top-10 |
| Quy trình (đã khóa) | (locked) protocol | Bộ thiết lập đánh giá cố định cho mọi run |
| Mô hình nền / giới hạn dưới | baseline / lower bound | Mức tối thiểu phương pháp khác phải vượt |
| Đối chứng | comparator | Phương pháp đem ra so sánh |
| So từng-nút-một | knob-for-knob | Chỉ đổi đúng MỘT thiết lập, giữ mọi thứ khác y hệt |
| Thí nghiệm đối chứng | ablation | Tắt/bật 1 thành phần để đo đóng góp của nó |
| Bị bác bỏ | falsified | Tuyên bố bị dữ liệu chứng minh là sai |
| Cơ chế | mechanism | Lý do *vì sao* một hiện tượng xảy ra |
| Đánh giá nguội / ấm | cold / warm eval | Chấm điểm bằng mô hình chưa-học / đã-học |
| Cổng tin cậy | trust gate | Điều kiện kiểm tra trước khi tin một con số |
| Seed | — | Hạt giống ngẫu nhiên để tái lập thí nghiệm |
| Tinh chỉnh cuối | final calibration | 1 lượt học ngắn căn lại phần cá nhân trước khi đánh giá |
| Đầu dân số | global / population head | Trung bình đầu tính điểm của mọi người dùng |
| Nhà tiên tri | oracle | Phép đo giả định "biết trước đáp án" — cho biết TRẦN của một ý tưởng |
| Kiểm tra chéo | cross-fitting | Chọn tham số trên nửa dân số, áp cho nửa kia (chống tự lừa) |
| Phản chứng | counterfactual | Đo "nếu KHÔNG có thành phần X thì mất bao nhiêu" |
| Cổng tái lập | replay trust gate | Bản chấm lại ngoại tuyến phải khớp từng-chữ-số với kết quả gốc |
| Kiểm định theo cặp | paired test (Wilcoxon) | So 2 hệ thống trên CÙNG từng người dùng, mạnh hơn so trung bình |

---

*Mọi con số được chép nguyên văn từ `final_metrics.best` trong `results.json` của từng module tại `results/federated/<module>/<run_id>/`; các phép đo phản chứng ở mục 8.3 từ các file `reeval_*.json` cùng thư mục (kèm phán quyết cổng tin cậy nhúng trong file); mốc centralized từ `results/centralized/bpr_mf_centralized_results.json`. Báo cáo này là bản giải thích của `THESIS_PROGRESS_REPORT.md` (bản 2026-07-27); mọi số liệu đã được kiểm chứng độc lập lại với JSON gốc ngày 2026-07-27.*
