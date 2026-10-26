---
name: vietnamese
description: This skill should be used when the user explicitly requests Vietnamese output — invokes `/vietnamese` (one reply) or `/vietnamese on` (rest of the session), says "answer in Vietnamese", "respond in Vietnamese", "in Vietnamese please", "trả lời bằng tiếng Việt", "dịch sang tiếng Việt", "viết bằng tiếng Việt" — or while persistence is ON from an earlier `on` command. Never activate merely because the user wrote in Vietnamese; an explicit request or an active ON state is required.
---

# Respond in Vietnamese

Two modes — pick one at activation:

- **Mode A — Respond in Vietnamese (default):** Claude's own replies are written in Vietnamese with the clarity rules below.
- **Mode B — Faithful translation:** the user hands over a specific text to translate ("dịch câu này", "dịch sát đoạn sau"). Fidelity wins: preserve structure, sentence boundaries, register, and tone; no glosses, no sentence-splitting, no unpacking. Notes about translation choices go outside the translated text, never inside it.

"Dịch sang tiếng Việt" pointing at a provided text → Mode B; meaning "reply to me in Vietnamese" → Mode A. When genuinely ambiguous, translate faithfully first and offer an explained version as a follow-up.

## Scope and precedence

- Vietnamese applies to **all user-visible prose in the turn**, including progress updates between tool calls. Internal reasoning, tool calls, and search queries stay in English.
- Files and artifacts follow the user's request and the deliverable's audience — never force a deliverable into English because of the internal-English rule. Code stays code; user-visible strings and comments follow the user's instruction.
- **The user's current explicit request beats every rule in this file** — including a request to translate a commit message or code strings, a request for brevity (then cut points, don't re-compress sentences), or a request to switch to another language.

## Activation and persistence

- `/vietnamese` → this reply only. `/vietnamese on` → every reply for the rest of the session. There is no off command: once ON, stay ON. The user writing a message in English is NOT an implicit off; only an explicit request for another language ends it (via precedence above).
- Never auto-activate just because the user wrote in Vietnamese.
- Meta-requests about this skill ("review skill vietnamese", "sửa description của skill này") are engineering tasks, not activation — answer in the conversation's current language.
- Persistence is a conversational convention read from chat history, not stored state; it never survives into a new session. If context compaction makes the state uncertain, ask once, briefly, in the language of the user's most recent message.

## Language rules (Mode A)

**Keep in English:** code, identifiers, paths, commands, commit messages, product names, acronyms (API, KYC, SQL), and domain jargon in technical registers — software (endpoint, deploy, race condition), betting and finance (margin, odds, stake, hedge), crypto, and similar. Vietnamese practitioners use these terms in English; translating them hurts readability. For **lay audiences** (medical, legal, everyday topics) prefer the standard Vietnamese term, optionally with the English term in parentheses.

**Gloss on first use:** the first time a technical term or acronym appears in a conversation, attach a short Vietnamese gloss in parentheses (3–10 words), using the meaning active in *this* conversation — polysemous terms (margin, CLV) never take a memorized default from another domain. Skip the gloss when the user already used the term fluently and unprompted; quoting someone else or asking "X là gì?" doesn't count as fluent usage. Never gloss inside quoted text or code. If compaction makes it uncertain whether a term was already glossed, a brief re-gloss beats skipping.

**Unpack compressed structures** — the most common failure mode:

- Name the measured quantity in every comparison clause; never "đắt gấp đôi A nhưng rẻ bằng nửa B" without saying *what* is compared (margin? odds? phí?). Repeat the noun even if it feels redundant.
- One quantitative idea per sentence; split chained comparisons and multi-condition rules into short sentences.
- Replace ambiguous pronouns and ellipses ("nó", "cái này", "mức đó") with the explicit noun.
- Numbers carry their labels on first mention in a paragraph ("margin ~2%", "odds 1.333"). Tables are exempt: the header row is the label.

**Style:** neutral professional tone ("tôi"/"bạn"). Clarity beats brevity. Block-level Markdown structure stays unchanged; sentence boundaries within prose may change under these rules — that is expected, not a violation.

## Examples

**Unpack (Mode A).** English draft: "Stake's football margin is ~4% — twice Pinnacle's but half of 1xbet's combo pricing."

- Wrong: "Margin bóng đá của Stake ~4% — đắt gấp đôi Pinnacle nhưng rẻ bằng nửa combo 1xbet."
- Right: "Margin (biên lợi nhuận nhà cái cài sẵn trong odds) của Stake với bóng đá khoảng 4%. Mức margin này cao gấp đôi margin ~2% của Pinnacle, nhưng chỉ bằng một nửa margin ~8% mà 1xbet thu trên kèo combo Result + Total."

**Faithful translation (Mode B).** Request: Dịch sát câu này: "The margin is thin, but it compounds."

- Right: "Biên lợi nhuận mỏng, nhưng nó tích luỹ dần."
- Wrong (Mode A rules misapplied): "Margin (biên lợi nhuận) của khoản này khá mỏng. Tuy nhiên, mức margin mỏng đó sẽ tích luỹ dần theo thời gian."