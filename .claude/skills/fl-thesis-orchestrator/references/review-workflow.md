# Review workflow script (track A, also usable for any document)

Pass the script inline to the `Workflow` tool with `args = { doc, dimensions, verifiers, now }`:

- `doc`: repo-relative path of the document to review.
- `dimensions`: `[{ key, prompt }]` — angles from the `methodology-review` skill, e.g.
  `{ key: 'leakage-and-protocol', prompt: 'Trace every negative pool, candidate list and seed; find any way the test item influences training or scoring.' }`,
  `{ key: 'statistics', prompt: 'Check the test, sidedness, zero handling, multiplicity, CI and effect-size choices against the registered rule.' }`,
  `{ key: 'comparability-and-asymmetry', prompt: 'List every difference between the compared runs/cells beyond the factor under study.' }`,
  `{ key: 'provenance-and-reproducibility', prompt: 'For each number and hash, find the file and key; flag anything that cannot be recomputed.' }`,
  `{ key: 'advisor-alignment', prompt: 'Compare the document with the advisor text quoted in it; flag deviations not declared in the implementer-decisions section.' }`,
  `{ key: 'writing', prompt: 'Terminology, diacritics, no infrastructure, tables before prose (advisor-facing drafts only).' }`.
- `verifiers`: 1 by default, 3 for "thorough" (majority rule).
- `now`: ISO timestamp from the main session (scripts cannot call `Date`).

```javascript
export const meta = {
  name: 'thesis-doc-review',
  description: 'Review a thesis document across angles, refute each finding, recompute its numbers',
  phases: [
    { title: 'Review', detail: 'one methodology-critic per angle' },
    { title: 'Verify', detail: 'refuters per finding; majority confirmed survives' },
    { title: 'Numbers', detail: 'numbers-auditor recomputes every figure' },
  ],
}

const FINDINGS = { type: 'object', required: ['findings'], properties: {
  findings: { type: 'array', items: { type: 'object',
    required: ['title', 'location', 'claim', 'evidence', 'severity', 'suggested_fix'], properties: {
      title: { type: 'string' }, location: { type: 'string' }, claim: { type: 'string' },
      evidence: { type: 'string' }, severity: { type: 'string', enum: ['blocking', 'major', 'minor'] },
      suggested_fix: { type: 'string' } } } } } }
const VERDICT = { type: 'object', required: ['status', 'reason'], properties: {
  status: { type: 'string', enum: ['confirmed', 'refuted', 'uncertain'] }, reason: { type: 'string' } } }
const MISMATCHES = { type: 'object', required: ['checked', 'mismatches', 'unsourced'], properties: {
  checked: { type: 'integer' },
  mismatches: { type: 'array', items: { type: 'object',
    required: ['location', 'stated', 'recomputed', 'source'], properties: {
      location: { type: 'string' }, stated: { type: 'string' }, recomputed: { type: 'string' }, source: { type: 'string' } } } },
  unsourced: { type: 'array', items: { type: 'object', required: ['location', 'stated'], properties: {
      location: { type: 'string' }, stated: { type: 'string' } } } } } }

const doc = args.doc
const verifiers = args.verifiers ?? 1
const dims = args.dimensions ?? []
if (!dims.length) return { error: 'no dimensions given', confirmed: [], dropped: [] }

// Numbers audit runs concurrently with the angle reviews; it does not depend on them.
const numbersP = agent(
  `Recompute every number in ${doc} from the result files it cites (results.json, manifest.json, reeval_*.json, stats_wilcoxon.json, per-user CSVs). Return data, not prose.`,
  { agentType: 'numbers-auditor', label: 'numbers', phase: 'Numbers', schema: MISMATCHES })

const reviewed = await pipeline(dims,
  d => agent(
    `Review ${doc} from the "${d.key}" angle: ${d.prompt}\nLoad the methodology-review skill first. Return findings as data; each must cite the section/line and the file+key you checked. Return an empty list if nothing holds up.`,
    { agentType: 'methodology-critic', label: `review:${d.key}`, phase: 'Review', schema: FINDINGS }),
  (r, d) => parallel((r?.findings ?? []).map(f => () =>
    parallel(Array.from({ length: verifiers }, (_, i) => () =>
      agent(
        `Try to REFUTE this finding about ${doc} (refuter ${i + 1} of ${verifiers}, angle ${d.key}). Re-read the cited location and files; recompute if a number is involved. Default to "refuted" when the evidence does not reproduce; "uncertain" only if the needed file is not on this machine. Finding: ${JSON.stringify(f)}`,
        { agentType: 'methodology-critic', label: `verify:${f.title}`, phase: 'Verify', schema: VERDICT })))
      .then(votes => {
        const ok = votes.filter(Boolean).filter(v => v.status === 'confirmed').length
        return { ...f, angle: d.key, votes: votes.filter(Boolean), confirmed: ok * 2 > verifiers }
      }))))

const lostAngles = reviewed.map((r, i) => (r === null ? dims[i].key : null)).filter(Boolean)
if (lostAngles.length) log(`${lostAngles.length} angle(s) lost: ${lostAngles.join(', ')}`)
const all = reviewed.flat().filter(Boolean)
const confirmed = all.filter(f => f.confirmed)
const numbers = await numbersP
log(`${all.length} findings, ${confirmed.length} confirmed with ${verifiers} refuter(s); numbers checked ${numbers?.checked ?? 'n/a'}, mismatches ${numbers?.mismatches?.length ?? 'n/a'}, unsourced ${numbers?.unsourced?.length ?? 'n/a'}`)
return { doc, reviewed_at: args.now ?? null, lostAngles, confirmed, dropped: all.filter(f => !f.confirmed), numbers }
```

Notes
- `pipeline` stage 2 receives `(prevResult, originalItem)`; the item gives the angle key for labels without threading it through stage 1.
- A thrown stage leaves `null` for that angle; `lostAngles` reports it and `.filter(Boolean)` keeps the rest alive.
- The script never calls `Date.now()`; the timestamp comes from `args.now`.
- To resume after editing the document, re-run with `resumeFromRunId`: angles whose prompt did not change return from cache, so only changed sections are re-reviewed when you edit the prompts to point at them.
