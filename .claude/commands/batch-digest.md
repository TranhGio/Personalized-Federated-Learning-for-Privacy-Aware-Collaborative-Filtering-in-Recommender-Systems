# Batch Digest All Unprocessed Papers

Scan `Papers/raw/` for all PDF files, then check which ones already have a corresponding digest in `Papers/digested/`. For each unprocessed paper, run the full digest workflow.

<task>
1. List all `.pdf` files in `Papers/raw/`
2. List all `.md` files in `Papers/digested/` (excluding `_INDEX.md`, `_TEMPLATE.md`, and `_EXAMPLE_*.md`)
3. For each PDF that does NOT have a corresponding digest:
   a. **Classify the paper**: Read the first 2 pages (title, abstract, section headings).
      - If the title/abstract contains words like "survey", "review", "overview", "taxonomy", "systematic study", "comprehensive study", OR the paper has 100+ references → it is a **SURVEY**
      - Otherwise → it is a **METHOD paper**
   b. Apply the correct digest workflow:
      - SURVEY → follow the template in `.claude/commands/digest-survey.md`
      - METHOD → follow the template in `.claude/commands/digest-paper.md`
   c. Save to `Papers/digested/<paper_id>.md`
   d. Update `Papers/digested/_INDEX.md`
4. At the end, print a summary: how many papers were processed, which were surveys vs methods, and any that failed

**Matching logic**: A PDF is considered "already digested" if any `.md` file in `Papers/digested/` references the paper's title or if the filename pattern matches. When uncertain, ask the user.

**Processing order**: Process papers in this priority order:
1. Survey papers (process these FIRST — they provide context for method papers)
2. Papers with "federated" + "recommend" in title/filename
3. Papers with "personalized" + "federated" in title/filename  
4. Papers with "privacy" + "federated" in title/filename
5. All remaining papers
</task>

<important>
- Do NOT skip any sections in the digest — follow the full template for the paper's type
- Survey papers are naturally longer to digest — this is expected
- For method papers longer than 20 pages, focus extraction on: Abstract, Method, Experiments, and Conclusion
- After all papers are processed, review `_INDEX.md` for consistency and update the Quick Stats
</important>
