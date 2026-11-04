# Chấm lại PFedRec (run cf72e0) có post-tuning bảng phim — cách (a)

**Trạng thái: ĐĂNG KÝ TRƯỚC, CHƯA CHẠY.** Phiên bản 1, ngày 04/11/2026.
Nguồn yêu cầu: góp ý của GVHD ngày 04/11/2026 — "(a) chấm lại ngoại tuyến từ run cf72e0 có post-tuning (chỉ dùng CPU,
tối đa 3 ngày)". Mọi lựa chọn dưới đây ghi trước khi chạy; kết quả điền vào `RESULT.md` cùng thư mục; README này
không sửa sau khi có số (thay đổi, nếu có, ghi vào mục 9 kèm ngày và lý do).

## 1. Câu hỏi

Số PFedRec hiện tại (NDCG@10 = 0.3447118) chấm mỗi user bằng **bảng phim chung** Q cộng head riêng đã hiệu chuẩn.
Code gốc (Zhangcx19/IJCAI-23-PFedRec, `fed_evaluate`) chấm mỗi user bằng **bảng phim riêng** của user đó sau epoch
cục bộ cuối (post-tuning). Câu hỏi: nếu chấm run cf72e0 theo kiểu post-tuning thì NDCG@10 bằng bao nhiêu, và phần
nào của khoảng cách tới số công bố 0.4436 (ML-1M, C = 1.0) là do cách chấm?

Kết quả dùng cho: (i) bảng kết quả PFedRec trong luận văn có thêm dòng "có post-tuning"; (ii) phần hạn chế nêu hai
nguyên nhân khác biệt với số công bố (C = 0.1 so với C = 1.0; không post-tuning khi chấm) — nguyên nhân thứ hai có
số đo. Không dùng để chọn phương pháp, không có ngưỡng QUA / KHÔNG QUA.

## 2. Đầu vào cố định

Giống hệt mục 2 của `../gate_qu/README.md` (v1.3), tóm lại:

| Hạng mục | Giá trị |
|---|---|
| Run nguồn | `20260726-083121-cf72e0`: cross-device, N = 6040, C = 0.1, 100 vòng, khôi phục vòng tốt nhất (83), hiệu chuẩn head 1 epoch, đánh giá D-06 ở chỉ số vòng 101 |
| Bảng chung Q | `global_state_best.pt`, khóa `embedding_item.weight` (3706 × 32), sha256 `279459ba72987a98dfd1065c59ab0bb32f0e6f33323c08ccbc25211103b41c57` |
| Head của user u | `affine_output.weight` từ cache `federated-pfedrec/.embedding_cache/20260726-083121-cf72e0/partition_{pid}.pt` (đã hiệu chuẩn) + `affine_output.bias` toàn cục trong `global_state_best.pt` |
| Tập ứng viên | D-06: phim test + 99 phim âm, giống hệt từng phim với lần chấm 0.3447118 (seed theo `round_num = 101`) |
| Mốc so sánh | `../reeval_cached_best_per_user.csv`: NDCG@10 0.3447118 (sparse 0.4269673 / medium 0.3801614 / dense 0.2935401), HR@10 0.6110927, MRR 0.2821309 |
| Nhóm user | theo số tương tác train, nửa mở: sparse [0, 30): 809; medium [30, 100): 2322; dense [100, ∞): 2909 |
| Nơi có đầu vào | `*.pt` và `.embedding_cache/` không có trên laptop; chạy trên máy đang giữ chúng, CPU. Trước khi chạy: `sha256sum` phải khớp, cache đủ `manifest.json` + 6040 file |

## 3. Giao thức post-tuning

Với mỗi user u (đủ 6040):

1. Khởi tạo Q_u ← Q (bảng chung tốt nhất), head (w_u, b) cố định.
2. Tinh chỉnh **chỉ các hàng của Q_u** trên dữ liệu train của u trong E epoch: BCE, 4 mẫu âm cho mỗi mẫu dương,
   batch 256, SGD không momentum, lr_item = 0.1 × 3706 × 80 = 29 648 (đúng tốc độ học hàng phim của run).
   Seed mẫu âm: `np_rng / torch_gen(42, pid, 200 + e)` như cổng Q_u (mục 3 của README cổng).
3. Chấm u bằng Q_u (α = 0, **không trộn** với Q) trên đúng 100 ứng viên D-06, hàm đánh giá của module.

Hai **tập mẫu âm** khi tinh chỉnh, chạy cả hai:

| Tên | Tập mẫu âm | Ý nghĩa |
|---|---|---|
| `clean` | {0..3705} \ T_u (phim test **nằm trong** tập) | không rò rỉ: client không biết phim test |
| `leaky` | {0..3705} \ (T_u ∪ {t_u}) | giao thức code gốc (`data.py` loại mọi phim đã tương tác, gồm phim test) = bảng loại trừ D-13 của ta |

Lưới: pool ∈ {clean, leaky} × E ∈ {1, 3} = **4 ô** (trùng D1–D4 ở mục 9 README cổng; E = 1 là `local_epoch`
mặc định của code gốc, E = 3 là ô phụ). E = 1 là ảnh chụp của quỹ đạo E = 3 (cùng seed). Không thêm ô sau khi chạy.

## 4. Khác biệt đã biết so với code gốc (ghi vào phần hạn chế)

| # | Code gốc | Ở đây | Hệ quả |
|---|---|---|---|
| K1 | Mỗi batch cập nhật xen kẽ head rồi bảng phim | head cố định, chỉ bảng phim | post-tuning của ta "yếu" hơn một bậc; head đã qua hiệu chuẩn nên khác biệt nhỏ |
| K2 | Bảng riêng là bảng sau vòng **tham gia cuối** của user (C = 1.0: luôn là vòng cuối) | mọi user tinh chỉnh từ Q của vòng tốt nhất 83 đã khôi phục | loại bỏ hiệu ứng head/bảng cũ do C = 0.1; đo đúng tác dụng của post-tuning |
| K3 | Có phim validation (thứ hai mới nhất) để chọn vòng | không có; vòng tốt nhất chọn trên phim test | đẩy số của ta lên (đã nêu ở phần hạn chế hiện tại) |
| K4 | Phim test bị loại khỏi mẫu âm | chạy cả hai (`leaky` = code gốc, `clean` = không rò rỉ) | tách được phần "rò rỉ" ra khỏi phần "post-tuning" |
| K5 | C = 1.0 | C = 0.1, không đổi | phần khoảng cách còn lại sau (a) ghi là "chưa tách", không gán cho C nếu không chạy thêm |

## 5. Quy tắc báo cáo (cố định trước khi chạy)

Bảng PFedRec trong luận văn có **ba dòng**, không chọn dòng cao nhất:

| Dòng | Ô | Nhãn trong luận văn |
|---|---|---|
| 1 | mốc 0.3447118 | PFedRec, chấm bằng bảng phim chung (shared item table) |
| 2 | `clean`, E = 1 | PFedRec, post-tuning, mẫu âm sạch |
| 3 | `leaky`, E = 1 | PFedRec, post-tuning, mẫu âm theo code gốc (loại phim test) |

Mỗi dòng: NDCG@10, HR@10, MRR, toàn bộ + 3 nhóm. Hai ô E = 3 vào phụ lục. Dòng 2 là con số được gọi là
"PFedRec có post-tuning" trong văn bản; dòng 3 chỉ dùng để giải thích khoảng cách với số công bố.

Câu trong phần hạn chế viết theo kết quả như sau:
- nếu Δ(dòng 3 − dòng 2) ≥ +0.010 và Wilcoxon một phía p < 0.05 → "một phần khoảng cách tới số công bố đến từ việc
  code gốc loại phim test khỏi mẫu âm khi post-tuning";
- ngược lại → "khác biệt về mẫu âm là không đáng kể; khoảng cách còn lại chưa tách được giữa C và các khác biệt K1–K3".

## 6. Kiểm định

- Mỗi ô so với mốc: Wilcoxon signed-rank theo cặp trên 6040 user, một phía (ô > mốc) là chính; hai phía, Pratt, Holm
  trên 4 ô, khoảng tin cậy 95% bootstrap (10 000 lần, seed 42), rank-biserial — đúng quy định mục 6 README cổng,
  dùng `scripts/thesis/gate_stats.py` (bỏ qua trường `verdict` GO/NO-GO, không áp dụng ngưỡng 0.3547).
- Hai phép đối chiếu theo cặp: `leaky` so với `clean` ở E = 1 và ở E = 3 (một phía, leaky > clean), cùng bộ
  Wilcoxon + bootstrap, toàn bộ và theo nhóm: cờ `--contrast <ô_A> <ô_B>` của `gate_stats.py`, ghi vào khóa
  `contrasts` của `stats_wilcoxon.json`.

## 7. Kiểm tra tin cậy (bắt buộc, hỏng là dừng, không ghi ô nào)

Cổng tin cậy `cached` của `reeval.py` (chấm lại mốc phải trùng từng user với CSV đã commit) chạy trước; sau đó:

| Kiểm tra | Điều kiện |
|---|---|
| Q_u hữu hạn | không NaN/Inf ở mọi ô (lr_item = 29 648) |
| Hàng chưa chạm | với 50 user đầu, hàng không bị rút làm mẫu âm/dương bằng đúng hàng của Q (bit-exact) |
| n_hit | `leaky`: số lần phim test bị rút = 0 ở mọi user; `clean`: > 0 trên toàn bộ |
| Đơn điệu ở `leaky` | head cố định, ứng viên chỉ có thể bị rút làm mẫu âm (điểm chỉ giảm), phim test không bị rút ⇒ NDCG_u(leaky) ≥ NDCG_u(mốc) với **mọi** user, ở cả E = 1 và E = 3. Vi phạm = lỗi cài đặt |
| Số user | 6040 ở mọi ô |

## 8. Dự đoán đăng ký trước

Lập luận cấu trúc như mục 8 README cổng: ứng viên là phim chưa xem, hàng của chúng chỉ đổi khi bị rút làm mẫu âm
và khi đó bị kéo xuống. Ở `clean`, phim test và mỗi phim âm có cùng xác suất bị rút nhưng tác dụng lên NDCG không
đối xứng ⇒ kỳ vọng Δ ≤ 0. Ở `leaky`, chỉ phim âm bị kéo xuống ⇒ Δ ≥ 0 theo từng user.

| # | Dự đoán | Cách kiểm |
|---|---|---|
| A1 | `clean`, E = 1: NDCG@10 toàn bộ **thấp hơn** mốc; kịch bản kéo mạnh (`mean_row_shift` ≫ ‖q_i‖): ≈ 0.321 (Δ ≈ −0.024); kịch bản kéo yếu: Δ ∈ [−0.005, 0]. E = 3 thấp hơn E = 1 | bảng 4 ô |
| A2 | `leaky`, E = 1: Δ so với mốc ≥ +0.010, p một phía < 0.05; E = 3 ≥ E = 1 | bảng 4 ô, Wilcoxon |
| A3 | Δ(`leaky` − `clean`) ≥ +0.010 ở cả E = 1 và E = 3, p < 0.05 | đối chiếu mục 6 |
| A4 | Nhóm dense có \|Δ\| lớn nhất ở cả 4 ô (nhiều tương tác ⇒ nhiều hàng ứng viên bị chạm: 27% ở E = 1 so với 2.5% sparse) | bảng theo nhóm |
| A5 | Không ô nào đạt 0.394 (= thu hẹp nửa khoảng cách tới 0.4436) | bảng 4 ô |

Nếu A1 đúng: "không post-tuning" **không** làm số của ta thấp hơn — ngược lại — và nguyên nhân thứ hai trong góp ý
của GVHD chỉ đúng khi chấp nhận rò rỉ phim test (A2, A3). Nếu A1 sai (clean > mốc): post-tuning có tác dụng thật và
dòng 2 trở thành số PFedRec mạnh hơn trong luận văn. Hai trường hợp đều ghi theo mục 5, không chọn lại.

## 9. Chạy và ghi kết quả

Chi phí: 2 quỹ đạo tinh chỉnh × 6040 user, đo trên dữ liệu tổng hợp cùng kích thước ≈ 3 phút CPU mỗi pool
(mục 11 README cổng); tổng dưới 15 phút, xa dưới giới hạn 3 ngày. GPU = 0.

Các bước, trên máy giữ đầu vào, nhánh `feat/gate-qu-reeval` sau khi merge:

```bash
# 0. (đã cài 04/11, cùng commit với README này) cờ --contrast trong gate_stats.py và kiểm tra "đơn điệu ở leaky"
#    (mục 7, counter `leaky_monotonicity_violations`, hỏng ⇒ thoát mã 3) trong run_qu_variant của reeval.py, có test.
# 1. tiền kiểm
sha256sum results/federated/pfedrec/20260726-083121-cf72e0/global_state_best.pt   # phải = 279459ba…57
test -f federated-pfedrec/.embedding_cache/20260726-083121-cf72e0/manifest.json
# 2. đo thử
python scripts/thesis/reeval.py --run-dir results/federated/pfedrec/20260726-083121-cf72e0 --module pfedrec \
  --globals best --local-variant qu_diag --device cpu --limit-users 200 --no-json --no-per-user
# 3. chạy đủ 4 ô
python scripts/thesis/reeval.py --run-dir results/federated/pfedrec/20260726-083121-cf72e0 --module pfedrec \
  --globals best --local-variant qu_diag --device cpu \
  --out-dir results/federated/pfedrec/20260726-083121-cf72e0/posttune
# 4. kiểm định
python scripts/thesis/gate_stats.py \
  --grid results/federated/pfedrec/20260726-083121-cf72e0/posttune/reeval_qu_diag_best_per_user.csv \
  --baseline results/federated/pfedrec/20260726-083121-cf72e0/reeval_cached_best_per_user.csv \
  --grid-json results/federated/pfedrec/20260726-083121-cf72e0/posttune/reeval_qu_diag_best.json \
  --contrast a0.00_E1_leaky a0.00_E1_clean --contrast a0.00_E3_leaky a0.00_E3_clean \
  --out results/federated/pfedrec/20260726-083121-cf72e0/posttune/stats_wilcoxon.json
# 5. điền RESULT.md (bảng 4 ô × toàn bộ + 3 nhóm × NDCG/HR/MRR; bảng Wilcoxon; đối chiếu A1–A5; commit, ngày), commit
```

Đầu ra: `reeval_qu_diag_best.json`, `reeval_qu_diag_best_per_user.csv` (15 cột như README cổng, `pool ∈ {clean, leaky}`),
`stats_wilcoxon.json`, `RESULT.md`. Lần chạy bị hủy ghi vào `RESULT.md` kèm lý do.

## 10. Nhật ký thay đổi

| Ngày | Phiên bản | Thay đổi |
|---|---|---|
| 2026-11-04 | 1 | Đăng ký trước theo góp ý GVHD 04/11 (cách (a)): 4 ô, quy tắc 3 dòng báo cáo, khác biệt K1–K5, kiểm tra tin cậy, dự đoán A1–A5 |
