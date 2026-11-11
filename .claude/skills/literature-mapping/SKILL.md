---
name: literature-mapping
description: "How to find, digest, index and map related work for this thesis: Papers/raw and Papers/digested layout, the digest templates in .claude/commands/, _INDEX.md maintenance, the protocol-caveat columns every published number must carry (participation ratio, post-tuning / own item table at inference, test-item exclusion from training negatives, validation split, negatives count, dataset version), mapping papers to the four contributions, and the current gap list. Use when the user says digest bài báo, tìm bài liên quan, related work, số công bố của paper X, so sánh với paper, or when a thesis claim needs a citation. Not for writing the Vietnamese related-work text itself (advisor-report / thesis-writer)."
---

# Literature mapping

The knowledge base exists so agents never re-read a PDF to answer the same question twice, and so no published number is quoted without the protocol that produced it.

## 1. Layout
- `Papers/raw/<Title-Author-Year>.pdf` — tracked in git (12 MB per batch is fine).
- `Papers/digested/<firstauthor>_<year>_<short>.md` — one digest per paper; `_INDEX.md` is the entry point (quick stats, topic sections, one-line summaries, last-updated date); `_EXAMPLE_mcmahan_2017_fedavg.md` shows the finished form.
- Templates: `.claude/commands/digest-paper.md` (method papers: core idea, problem, method with objective/algorithm/hyper-parameters, implementation notes, results with datasets and key tables, connections to the thesis, references to follow) and `digest-survey.md` (surveys: taxonomy, positioning of our methods, open problems). Classify first: "survey / review / taxonomy" in the title or 100+ references → survey template.

## 2. Procedure for a new paper
1. Check `_INDEX.md` and `ls Papers/digested` for an existing digest (also under a different short name).
2. Obtain the PDF (publisher, arXiv, author page); save to `Papers/raw/`; record DOI/arXiv id in the digest header.
3. Fill the template from the paper's own tables; cite table numbers; keep the metric names the paper uses and add the mapping to ours (HR@10 / NDCG@10 on ML-1M sampled-99 ≈ our `sampled_hr@10` / `sampled_ndcg@10`).
4. Add the protocol row (section 3) under "Experimental Results".
5. Update `_INDEX.md`: total count, topic section line, last-updated date.
6. Return the digest path and the three facts most relevant to the question that triggered the search.

## 3. Protocol columns for any published number
| Column | Why it matters here |
|---|---|
| Dataset + version, split rule, validation item (yes/no) | we have no validation item; best round chosen on the test item |
| Participation ratio C per round | ours 0.1; PFedRec paper 1.0 |
| Inference table: shared global vs own post-tuned | 0.3447 (shared) vs 0.3789 leaky E=1 / 0.4520 leaky E=3 on cf72e0 |
| Training negatives exclude the test item? | official PFedRec `data.py`: yes (leak); our clean pool: no |
| Eval negatives count and sampling seed policy | 99, seeded by round |
| Local epochs, lr for item rows, batch | PFedRec: 1 epoch, lr 0.1 × 3706 × 80 for items |
| Rounds, early stopping, checkpoint rule | ours 100 rounds, best_round_restore |
| Reported statistic | mean over users? seeds? any test? |

A table that lacks a column says "không rõ" in that cell rather than leaving it blank.

## 4. Mapping to the four contributions
1. **Where should personalization live** — FedPer, FedRep, APFL, FedRecon, PFedRec, FedRAP, GPFedRec, FedTHE; record what each keeps local and what is averaged.
2. **Adaptive as negative result** — APFL (learned α), FedTHE (per-sample ensemble weight), FedCA (embedding skew); what they report when α pulls toward the global.
3. **Privacy with numbers** — embedding-inversion / preference reconstruction from user vectors, secure aggregation, DP for FedRec; byte-level communication accounting in FedRec papers.
4. **Valid evaluation for stateful clients** — stateless vs stateful client evaluation (FedRecon), post-tuning at inference (PFedRec), leakage in negative sampling (NCF-protocol critiques).

## 5. Known gaps (2026-11)
- FedDAE (AAAI 2025) requested by the advisor — not digested, PDF not in `Papers/raw/`.
- No digest yet on preference-reconstruction attacks from user embeddings (needed for contribution 3).
- CoFedRec SOTA numbers (HR@10 77.75 / NDCG@10 48.81, ML-1M) are in the index without a protocol row.

## 6. Output for the thesis writer
`_workspace/<track>/NN_literature-scout_<topic>.md`: ranked list (paper, venue, year, what it contributes to which contribution, protocol row), then a comparison table ready to be translated; English is fine here, the writer translates.
