# Kết quả và quyết định cổng Q_u (điền SAU khi chạy)

**Trạng thái: CHƯA CHẠY.** Mẫu này được tạo cùng README phiên bản 1.1 (03/11/2026) để kết quả chỉ được điền vào
các ô đã định sẵn. Không thêm chỉ số, ô hay tiêu chí mới sau khi xem số liệu.

| Hạng mục | Giá trị |
|---|---|
| Ngày chạy | — |
| Commit mã nguồn | — |
| Commit README được dùng | — |
| sha256 `global_state_best.pt` | — (phải = `279459ba…57`) |
| Cổng tin cậy `cached` | — (PASS, max\|Δ\|) |
| Ô α = 1.0 trùng bit điểm từng user | — (E = 1: …/6040; E = 3: …/6040) |
| Thời gian đo trước (`--limit-users 200`, CPU) | — |
| Thời gian tinh chỉnh toàn bộ (CPU) | — |
| Số lần chạy bị hủy vì điều kiện tin cậy | — (mỗi lần: ngày, kiểm tra hỏng, commit sửa, có đổi mã tinh chỉnh/chấm điểm không) |

## 1. Bảng 8 ô (NDCG@10 / HR@10; mốc 0.3447118 / 0.6110927)

| Ô | α | E | NDCG@10 toàn bộ | Δ | HR@10 toàn bộ | NDCG sparse | NDCG medium | NDCG dense |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.3 | 1 | — | — | — | — | — | — |
| 2 | 0.5 | 1 | — | — | — | — | — | — |
| 3 | 0.7 | 1 | — | — | — | — | — | — |
| 4 | 1.0 | 1 | — | — | — | — | — | — |
| 5 | 0.3 | 3 | — | — | — | — | — | — |
| 6 | 0.5 | 3 | — | — | — | — | — | — |
| 7 | 0.7 | 3 | — | — | — | — | — | — |
| 8 | 1.0 | 3 | — | — | — | — | — | — |

## 2. Kiểm định Wilcoxon theo từng user (so với mốc; một phía theo README mục 12 điểm 7, p hai phía báo cáo kèm; 6 ô có α < 1)

| Ô | n_nonzero | p một phía (wilcox) | p hai phía | p (pratt) | p Holm | r rank-biserial | Δ NDCG trung bình [CI 95%] | Δ HR trung bình [CI 95%] |
|---|---|---|---|---|---|---|---|---|
| 1 | — | — | — | — | — | — | — | — |
| 2 | — | — | — | — | — | — | — | — |
| 3 | — | — | — | — | — | — | — | — |
| 5 | — | — | — | — | — | — | — | — |
| 6 | — | — | — | — | — | — | — | — |
| 7 | — | — | — | — | — | — | — | — |

## 3. Đối chiếu dự đoán đăng ký trước

| # | Dự đoán (README mục 8; P6 mục 9) | Kết quả | Đúng / Sai |
|---|---|---|---|
| P1 | — | — | — |
| P2 | — | — | — |
| P3 | — | — | — |
| P4 | — | — | — |
| P5 | — | — | — |
| P6 (chỉ khi chạy biến thể chẩn đoán) | — | — | — |
| P7 | — | — | — |

## 4. Biến thể chẩn đoán D1–D4 (chỉ khi được duyệt; không dùng cho quyết định; tách khỏi bảng 8 ô)

| Biến thể | Tập mẫu âm | E | NDCG@10 | HR@10 | Ghi chú |
|---|---|---|---|---|---|
| D1 | sạch | 1 | — | — | — |
| D2 | rò rỉ | 1 | — | — | — |
| D3 | sạch | 3 | — | — | — |
| D4 | rò rỉ | 3 | — | — | — |

## 5. Kết luận

- Điều kiện 1 (có ô ≥ 0.3547): —
- Điều kiện 2 (Wilcoxon p < 0.05 cho ô đó): —
- Ô thắng (nếu QUA, theo README 7.1 điểm 3; nhãn "ước lượng Giai đoạn 1, chưa xác nhận"): —
- **Quyết định: QUA CỔNG / KHÔNG QUA CỔNG** — 
- Giải thích ngắn (≤ 5 dòng): —
