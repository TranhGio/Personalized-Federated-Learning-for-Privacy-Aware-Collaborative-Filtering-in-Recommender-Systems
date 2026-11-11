# Trigger cases for `fl-thesis-orchestrator` (kept for `harness:evolve`)

## Should trigger (→ track)
1. "Viết README đăng ký trước cho thí nghiệm đo chi phí head bị cũ" → A
2. "Review lại README posttune cho kỹ trước khi chạy" → A (review only, thorough)
3. "Kết quả posttune về rồi, điền RESULT.md" → D
4. "So sánh 4 module trên NDCG@10 theo nhóm, có kiểm định theo cặp" → D (+B if adapters missing)
5. "Soạn báo cáo thứ Sáu cho cô" → E
6. "Cô vừa gửi góp ý mới, phân tích và trả lời" → E
7. "Viết mục hạn chế cho chương kết quả" → E
8. "Cài adapter Baseline cho reeval.py theo README" → B
9. "Thêm cờ --contrast vào gate_stats và test" → B
10. "Digest FedDAE và map vào đóng góp 1" → F
11. "Chạy lại phản biện sau khi tôi sửa mục 6" → A (resume)
12. "Cập nhật tóm tắt tình hình cho cô với số mới" → E

## Should NOT trigger (near misses → what instead)
1. "Chạy posttune trên máy train" → `train-remote` skill
2. "Tạo sweep W&B cho adaptive" → `sweep` skill
3. "NDCG của cf72e0 là bao nhiêu?" → read `results.json` directly
4. "Sửa lỗi typing Python 3.9 trong task.py" → plain code edit (no experiment dependency)
5. "Dịch đoạn này sang tiếng Việt" → `vietnamese` skill / direct
6. "Commit và mở PR" → git directly
7. "Giải thích FedProx cho tôi" → direct answer (maybe `Papers/digested`)
8. "Tóm tắt file Vinh_lv.docx" → direct read (track E only if a reply/report is wanted)
9. "Xoá cache cũ trên máy trạm" → `train-remote` / remote scripts
10. "Kiểm tra PR #11 có cần merge không" → git/gh directly
