# Kết quả chấm lại PFedRec có post-tuning — cách (a)

**Trạng thái: ĐÃ CHẠY, 04/11/2026.** Điền theo mẫu tạo cùng README phiên bản 1 (04/11/2026); không thêm chỉ số, ô
hay tiêu chí sau khi xem số liệu. Nguồn số: `reeval_qu_diag_best.json` (`metrics_by_cell`, `trust_checks`),
`stats_wilcoxon.json` (`cells`, `contrasts`). Chạy bằng wrapper `.bug-investigation/run_pfedrec_posttune_a.sh`
qua `scripts/remote/start_run.sh`, CPU, log `posttune-a`.

Bản sửa 2026-11-11 sau phản biện thống kê (F1–F9 trong `_workspace`): bổ sung các cột đã đăng ký (Pratt, HR, CI theo
nhóm), sửa diễn giải A1, ghi nguồn; không đổi số liệu ô, quy tắc hay kết luận mục 5.

| Hạng mục | Giá trị |
|---|---|
| Ngày chạy | 2026-11-04, 21:57–22:02 (theo log `posttune-a.log` trên máy chạy, không có trong repo) |
| Commit mã nguồn (máy chạy) | `d30ce88` (= `master` sau PR #16, #17) |
| Commit README được dùng | `67835a2` (README v1.1, merged in PR #16) |
| sha256 `global_state_best.pt` | `279459ba72987a98dfd1065c59ab0bb32f0e6f33323c08ccbc25211103b41c57` (khớp) |
| Cổng tin cậy `cached` | PASS, max\|Δ\| = 0.0 (NDCG/HR toàn bộ và 3 nhóm trùng số của run; 6040/809/2322/2909 user) |
| Kiểm tra tin cậy mục 7 (`trust_checks.status`) | PASS: `leaky_monotonicity_violations` = {E1: 0, E3: 0}; `n_hit.leaky` = 0; `n_hit.clean` = 3618; Q_u không NaN/Inf; hàng chưa chạm trùng bit (50 user); bất biến d_u: 0 vi phạm ở 4 ô; lỗi danh sách ứng viên: 0; lr_item = 29 648 |
| Thời gian: 200 user / toàn bộ | không ghi mốc riêng cho bước dò (log không in) / 292.3 s cho 4 ô × 6040 user (≈ 20.7 user/s, mỗi user gồm cả 4 ô) + 3.8 s chấm lại mốc (theo log); bước 2 từ 21:57:03 đến 22:02:01 (theo log `posttune-a.log` trên máy chạy, không có trong repo; JSON chỉ có `elapsed_sec` = 292.27) |
| Trung vị `mean_row_shift` so với trung vị ‖q_i‖ | 16.32 (E1) và 13.55 (E3) so với 12.28 → tỉ số ≈ 1.3 / 1.1: **không** phải kéo mạnh (≫ 10×) theo định nghĩa mục 8, cũng không phải kéo yếu (≪ 1) |
| Lần chạy bị hủy | 0 (lần đầu `start_run.sh` dừng trước khi chạy vì nhánh `master` trên máy chưa có upstream; đặt `--set-upstream-to` rồi chạy lại, không có ô nào được tính; theo log `posttune-a.log` trên máy chạy, không có trong repo) |

## 1. Bốn ô (NDCG@10 / HR@10 / MRR, toàn bộ và theo nhóm)

| Ô | Toàn bộ | sparse (809) | medium (2322) | dense (2909) |
|---|---|---|---|---|
| Mốc: bảng phim chung | 0.3447118 / 0.6110927 / 0.2821309 | 0.4269673 / 0.7058096 / 0.3551956 | 0.3801614 / 0.6688200 / 0.3082912 | 0.2935401 / 0.5386731 / 0.2409299 |
| `a0.00_E1_clean` (dòng 2 luận văn) | 0.3181 / 0.5584 / 0.2608 | 0.4165 / 0.6848 / 0.3474 | 0.3611 / 0.6326 / 0.2937 | 0.2564 / 0.4641 / 0.2105 |
| `a0.00_E1_leaky` (dòng 3 luận văn) | 0.3789 / 0.6642 / 0.3090 | 0.4305 / 0.7108 / 0.3583 | 0.3891 / 0.6792 / 0.3165 | 0.3564 / 0.6394 / 0.2893 |
| `a0.00_E3_clean` (phụ lục) | 0.2762 / 0.4704 / 0.2305 | 0.3981 / 0.6539 / 0.3329 | 0.3347 / 0.5792 / 0.2742 | 0.1956 / 0.3324 / 0.1672 |
| `a0.00_E3_leaky` (phụ lục) | 0.4520 / 0.7505 / 0.3738 | 0.4385 / 0.7219 / 0.3649 | 0.4128 / 0.7063 / 0.3381 | 0.4871 / 0.7937 / 0.4047 |

## 2. Wilcoxon theo cặp so với mốc (NDCG@10 và HR@10)

Nguồn: `stats_wilcoxon.json` → `cells.<ô>.by_scope.<nhóm>` (`mean_delta_ndcg`, `ci95_ndcg`, `mean_delta_hr`, `ci95_hr`,
`p_one_sided`, `p_two_sided`, `p_pratt_one_sided`, `p_holm_one_sided`, `rank_biserial_r`, `w_plus`, `w_minus`);
n₊ / n₋ / n₀ = số user có d_u > 0 / < 0 / = 0 (d_u = NDCG của ô − NDCG mốc), tính lại từ
`reeval_qu_diag_best_per_user.csv` và `reeval_cached_best_per_user.csv`. "p một phía" là kiểm định đăng ký
(ô > mốc, `zero_method=wilcox`); "p Pratt" là kiểm tra độ nhạy đã đăng ký (cùng chiều, `zero_method=pratt`).

| Ô | Δ NDCG@10 [CI95 bootstrap] | Δ HR@10 [CI95] | n₊ / n₋ / n₀ | n≠0 | p một phía (wilcox) | p hai phía | p Pratt (một phía) | p Holm (một phía, 4 ô) | r rank-biserial |
|---|---|---|---|---|---|---|---|---|---|
| `a0.00_E1_clean` | −0.02664 [−0.03153, −0.02173] | −0.0526 [−0.0611, −0.0440] | 979 / 516 / 4545 | 1495 | 1.0 | 9.5e−4 | 1.4e−22 | 1.0 | −0.099 |
| `a0.00_E1_leaky` | +0.03421 [+0.03184, +0.03661] | +0.0531 [+0.0475, +0.0589] | 1329 / 0 / 4711 | 1329 | p < 1e−10 (W⁻ = 0 theo cấu trúc) | < 1e−10 (cấu trúc) | < 1e−10 (cấu trúc) | < 1e−10 (cấu trúc) | +1.000 |
| `a0.00_E3_clean` | −0.06850 [−0.07570, −0.06131] | −0.1407 [−0.1525, −0.1286] | 1281 / 1160 / 3599 | 2441 | 1.0 | 3.0e−53 | 0.996 | 1.0 | −0.359 |
| `a0.00_E3_leaky` | +0.10733 [+0.10256, +0.11218] | +0.1394 [+0.1308, +0.1482] | 2502 / 0 / 3538 | 2502 | p < 1e−10 (W⁻ = 0 theo cấu trúc) | < 1e−10 (cấu trúc) | < 1e−10 (cấu trúc) | < 1e−10 (cấu trúc) | +1.000 |

Ghi chú:
- p Holm trong bảng là Holm trên p **một phía** của 4 ô (`gate_stats.py`). Holm trên p **hai phía** (tính lại bằng
  `holm_adjust`): `E1_clean` 9.5e−4, `E1_leaky` 1.7e−218, `E3_clean` 6.1e−53, `E3_leaky` < 1e−300 (số học tràn về 0.0); không
  kết luận nào đổi.
- Hai ô `leaky`: p-value là hệ quả cấu trúc của kiểm tra đơn điệu (W⁻ = 0, r = +1.000, không user nào giảm); các chữ số
  như 2.9e−219 (một phía) hay 5.8e−219 (hai phía) chỉ là đuôi của xấp xỉ chuẩn, không mang thông tin thêm. Bằng chứng
  quyết định nằm ở Δ và CI (xem mục 5).
- Ô `clean`: trung bình (CI loại 0) và kiểm định hạng không cùng chiều. Với `E1_clean` toàn bộ: p một phía 0.9995 và p hai
  phía 9.5e−4 (W⁺ = 503 966 < W⁻ = 614 294), nhưng p Pratt một phía = 1.4e−22 (ô > mốc). Lý do: d_u trộn nhiều mức tăng
  nhỏ (n₊ = 979) với ít mức giảm lớn (n₋ = 516; cả 516 user này có NDCG = 0 đúng và phim test bị rút, `test_row_touched` = 1),
  nên trung bình giảm trong khi hạng của phần lớn user đổi chiều lên. Đảo chiều tương tự ở `E1_clean` medium và dense,
  `E3_clean` sparse và medium (bảng dưới).

### Theo nhóm (Δ NDCG@10; Δ HR@10 trong ngoặc)

| Ô | Nhóm | Δ NDCG@10 | CI95 | Δ HR@10 [CI95] | p một phía | p hai phía | p Pratt (một phía) | n₊ / n₋ / n₀ |
|---|---|---|---|---|---|---|---|---|
| `E1_clean` | sparse | −0.0105 | [−0.0181, −0.0039] | −0.0210 [−0.0334, −0.0099] | 0.96 | 0.090 | 0.12 | 30 / 21 / 758 |
| `E1_clean` | medium | −0.0191 | [−0.0250, −0.0133] | −0.0362 [−0.0457, −0.0267] | 0.64 | 0.72 | 1.1e−11 | 246 / 108 / 1968 |
| `E1_clean` | dense | −0.0371 | [−0.0460, −0.0286] | −0.0746 [−0.0904, −0.0591] | 0.999 | 1.7e−3 | 4.9e−11 | 703 / 387 / 1819 |
| `E3_clean` | sparse | −0.0289 | [−0.0414, −0.0176] | −0.0519 [−0.0705, −0.0334] | 0.97 | 0.059 | 4.2e−3 | 91 / 53 / 665 |
| `E3_clean` | medium | −0.0454 | [−0.0550, −0.0363] | −0.0896 [−0.1047, −0.0745] | 1.0 | 5.7e−4 | 9.3e−9 | 507 / 279 / 1536 |
| `E3_clean` | dense | −0.0979 | [−0.1099, −0.0858] | −0.2063 [−0.2265, −0.1853] | 1.0 | 2.0e−50 | 1.0 | 683 / 828 / 1398 |
| `E1_leaky` | sparse | +0.0036 | [+0.0018, +0.0058] | +0.0049 [+0.0012, +0.0099] | 2.6e−7 (W⁻ = 0 theo cấu trúc, chỉ 33 user đổi) | 5.2e−7 | 4.6e−9 | 33 / 0 / 776 |
| `E1_leaky` | medium | +0.0090 | [+0.0073, +0.0108] | +0.0103 [+0.0065, +0.0146] | p < 1e−10 (W⁻ = 0 theo cấu trúc) | < 1e−10 | < 1e−10 | 262 / 0 / 2060 |
| `E1_leaky` | dense | +0.0629 | [+0.0583, +0.0675] | +0.1007 [+0.0901, +0.1117] | p < 1e−10 (W⁻ = 0 theo cấu trúc) | < 1e−10 | < 1e−10 | 1034 / 0 / 1875 |
| `E3_leaky` | sparse | +0.0115 | [+0.0081, +0.0152] | +0.0161 [+0.0074, +0.0260] | p < 1e−10 (W⁻ = 0 theo cấu trúc) | < 1e−10 | < 1e−10 | 98 / 0 / 711 |
| `E3_leaky` | medium | +0.0326 | [+0.0294, +0.0361] | +0.0375 [+0.0297, +0.0457] | p < 1e−10 (W⁻ = 0 theo cấu trúc) | < 1e−10 | < 1e−10 | 629 / 0 / 1693 |
| `E3_leaky` | dense | +0.1936 | [+0.1850, +0.2020] | +0.2551 [+0.2399, +0.2712] | p < 1e−10 (W⁻ = 0 theo cấu trúc) | < 1e−10 | < 1e−10 | 1775 / 0 / 1134 |

Ở mọi nhóm của hai ô `clean`, CI95 bootstrap của Δ NDCG@10 loại 0 (đều âm), trong khi p hai phía của sparse (0.090, 0.059)
và medium của `E1_clean` (0.72) không có ý nghĩa. Hai thước đo đo hai thứ khác nhau: CI là của **trung bình** d_u, bị kéo
bởi một số ít user rơi về 0; kiểm định hạng nhạy với dấu của phần đông user, phần đông này lại tăng nhẹ. Không đọc các p
hai phía đó như "không giảm". Nhóm sparse `E1_clean` là nhóm duy nhất mà cả ba p (0.96 / 0.090 / 0.12) đều không ủng hộ
một chiều rõ ràng, trong khi CI vẫn loại 0. r = +1.000 ở hai ô `leaky` là hệ quả của kiểm tra đơn điệu (không user nào giảm).

## 3. Đối chiếu leaky − clean (NDCG@10)

Nguồn: `stats_wilcoxon.json` → `contrasts.<A>__minus__<B>.by_scope.<nhóm>`; n₊ / n₋ / n₀ tính lại từ CSV. Ở mọi dòng
n₋ = 0 (W⁻ = 0) theo cấu trúc, nên p-value dưới đây là "p < 1e−10 (W⁻ = 0 theo cấu trúc)"; bằng chứng là Δ và CI.

| Đối chiếu | Nhóm | Δ NDCG@10 | CI95 | p một phía | p hai phía | p Pratt (một phía) | n₊ / n₋ / n₀ |
|---|---|---|---|---|---|---|---|
| E = 1: leaky − clean | toàn bộ | +0.06085 | [+0.05583, +0.06578] | p < 1e−10 (W⁻ = 0 theo cấu trúc) | < 1e−10 | < 1e−10 | 639 / 0 / 5401 |
| E = 1: leaky − clean | sparse | +0.0141 | [+0.0078, +0.0213] | 2.8e−5 | 5.5e−5 | 2.3e−6 | 21 / 0 / 788 |
| E = 1: leaky − clean | medium | +0.0281 | [+0.0225, +0.0338] | p < 1e−10 (W⁻ = 0 theo cấu trúc) | < 1e−10 | < 1e−10 | 108 / 0 / 2214 |
| E = 1: leaky − clean | dense | +0.1000 | [+0.0913, +0.1090] | p < 1e−10 (W⁻ = 0 theo cấu trúc) | < 1e−10 | < 1e−10 | 510 / 0 / 2399 |
| E = 3: leaky − clean | toàn bộ | +0.17583 | [+0.16774, +0.18384] | p < 1e−10 (W⁻ = 0 theo cấu trúc) | < 1e−10 | < 1e−10 | 1694 / 0 / 4346 |
| E = 3: leaky − clean | sparse | +0.0404 | [+0.0295, +0.0524] | p < 1e−10 (W⁻ = 0 theo cấu trúc) | 9.2e−11 | 6.1e−14 | 55 / 0 / 754 |
| E = 3: leaky − clean | medium | +0.0781 | [+0.0692, +0.0875] | p < 1e−10 (W⁻ = 0 theo cấu trúc) | < 1e−10 | < 1e−10 | 295 / 0 / 2027 |
| E = 3: leaky − clean | dense | +0.2915 | [+0.2780, +0.3046] | p < 1e−10 (W⁻ = 0 theo cấu trúc) | < 1e−10 | < 1e−10 | 1344 / 0 / 1565 |

Ghi chú cơ chế: hai quỹ đạo `clean` và `leaky` chỉ khác nhau ở những lượt rút trúng phim test, nên hiệu leaky − clean
khác 0 ở đúng 639 user (E = 1) và 1694 user (E = 3). Δ toàn bộ +0.061 dồn vào 639 user tương ứng ≈ +0.58 NDCG mỗi user
bị ảnh hưởng: khi phim test bị rút làm mẫu âm dù chỉ một lần, NDCG của user đó gần như về 0.

## 4. Đối chiếu dự đoán A1–A5 (README mục 8)

| # | Dự đoán | Kết quả | Đúng / Sai / Không phân biệt |
|---|---|---|---|
| A1 | clean E1 < mốc; kéo mạnh ≈ 0.321 hoặc kéo yếu Δ ∈ [−0.005, 0]; E3 < E1 | clean E1 = 0.3181 (Δ −0.0266, CI [−0.0315, −0.0217]; dự đoán kéo mạnh −0.024, nằm trong CI); E3 = 0.2762 (Δ −0.0685, CI [−0.0757, −0.0613]; dự đoán −0.061, nằm ngoài CI, lệch 0.0075 > biên đăng ký 0.001) < E1. Nguồn −0.024 / −0.061: `gate_qu/README.md` mục 8, P1 (cận xấu nhất −0.032 / −0.074); không có trong `posttune/README.md`. p Pratt một phía của `E1_clean` toàn bộ = 1.4e−22 (xem mục 2). Chỉ số `mean_row_shift` lại không ở mức "kéo mạnh" (≈ 1.3× ‖q_i‖) | **Đúng** về chiều **của trung bình** (CI loại 0; E3 < E1). Độ lớn: khớp ở E1; ở E3 mức giảm lớn hơn dự đoán (−0.061 ngoài CI), vẫn trong cận xấu nhất −0.074 của `gate_qu/README.md`. Dưới cách xử lý số 0 kiểu Pratt, kiểm định hạng lại chỉ chiều ngược lại, vì đa số user thay đổi tăng nhẹ còn 516 user (E1) có phim test bị rút giảm về 0. Nhận diện kịch bản bằng `mean_row_shift` không phân biệt được |
| A2 | leaky E1: Δ ≥ +0.010, p < 0.05; E3 ≥ E1 | Δ +0.0342 [+0.0318, +0.0366], p < 1e−10 (W⁻ = 0 theo cấu trúc); E3 (0.4520) > E1 (0.3789) | **Đúng** |
| A3 | leaky − clean ≥ +0.010, p < 0.05, ở cả E1 và E3 | +0.0609 [+0.0558, +0.0658] và +0.1758 [+0.1677, +0.1838], cả hai p < 1e−10 (W⁻ = 0 theo cấu trúc) | **Đúng** |
| A4 | dense có \|Δ\| lớn nhất ở cả 4 ô | E1_clean 0.037 > 0.019 > 0.011; E3_clean 0.098 > 0.045 > 0.029; E1_leaky 0.063 > 0.009 > 0.004; E3_leaky 0.194 > 0.033 > 0.012 | **Đúng** |
| A5 | không ô nào ≥ 0.394 | `a0.00_E3_leaky` = 0.4520 ≥ 0.394, và vượt cả số công bố 0.4436; `E1_leaky` = 0.3789 < 0.394. Nguồn 0.4436 và 0.394: `posttune/README.md` mục 1 và mục 8 (A5) | **Sai** (ở E = 3) |

## 5. Câu cho phần hạn chế (chọn theo quy tắc mục 5 README, không viết lại)

- [x] Δ(dòng 3 − dòng 2) = +0.0609 ≥ +0.010 và p < 1e−10 (W⁻ = 0 theo cấu trúc) < 0.05 → "một phần khoảng cách tới số công bố
  đến từ việc code gốc loại phim test khỏi mẫu âm khi post-tuning".
  (Điều kiện p thỏa theo cấu trúc; bằng chứng quyết định là Δ = +0.0609 với CI95 [+0.0558, +0.0658], cận dưới ≥ +0.010.)
- [ ] ngược lại → (không áp dụng).

Bảng PFedRec trong luận văn theo mục 5 README: dòng 1 = 0.3447 (bảng phim chung), dòng 2 = 0.3181 (post-tuning, mẫu âm
sạch), dòng 3 = 0.3789 (post-tuning, mẫu âm theo code gốc). Dòng 2 là số được gọi là "PFedRec có post-tuning".
Hai ô E = 3 ở phụ lục; `E3_leaky` = 0.4520 chỉ dùng để minh họa rằng giao thức post-tuning của code gốc, với 3 epoch,
tái tạo và vượt số công bố ngay ở C = 0.1 — không trích làm kết quả phương pháp.

Người điền: Claude (từ hai JSON), kiểm tra lại: Dang Vinh | Ngày: 2026-11-04 | Commit kết quả: `455b12d`
