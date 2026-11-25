# Thesis (LaTeX)

Master's thesis, FPT University, Master of Software Engineering. Layout follows the example
theses in `../thesis_examples/` (A4, Times-like serif, two cover pages, abstract page with the
degree line, acknowledgements, ToC, lists, numbered references with DOIs, appendices).

## Build

Preferred engine: XeLaTeX (as the 2026 example thesis); pdfLaTeX also works (newtx fonts).
Bibliography: biblatex + biber.

```bash
# 1. regenerate every table, figure and number macro from the result files
python thesis/scripts/build_assets.py          # run from the repository root

# 2. build
cd thesis && latexmk -xelatex main.tex          # or: latexmk -pdf main.tex
```

On Overleaf: upload the `thesis/` folder, set the compiler to XeLaTeX and the main document
to `main.tex`. The required TeX Live packages are standard (fontspec, unicode-math, biblatex,
booktabs, tikz, titlesec, tocloft, fancyhdr, longtable, caption, subcaption, enumitem).

Local install on Debian/Ubuntu if no TeX is present:

```bash
sudo apt install texlive-xetex texlive-latex-extra texlive-fonts-extra texlive-bibtex-extra \
                 texlive-science biber latexmk
```

## Layout

```
main.tex                  document skeleton and metadata (title, author, supervisor, year)
fptmse.sty                page layout, fonts, cover pages, placeholder macros
frontmatter/              abstract, acknowledgements, abbreviations, notation
chapters/01..07_*.tex     introduction, literature, methodology, setup, results, discussion, conclusion
appendices/A..C_*.tex     post-tuning tables, configurations, reproducibility
tables/                   GENERATED: tab_*.tex and numbers.tex (one macro per cited number)
figures/                  GENERATED fig_*.pdf, plus the hand-written TikZ fig_placement.tex
scripts/build_assets.py   generator; scripts/lint_tex.py structural checks without a TeX install
references.bib            bibliography (entries marked "TODO verify" need a DOI check)
```

## Rules

- No number is typed into the `.tex` files by hand. Prose uses the macros of
  `tables/numbers.tex`; tables are `\input`. If a number is missing, add it to
  `build_assets.py` and regenerate.
- `\pending{...}` marks a result that has not been produced yet; `\decision{...}` marks a
  wording or scope choice the advisor has to make. Both print as orange boxes. Uncomment
  `\finalmodetrue` in `main.tex` for the submission build: every remaining placeholder then
  prints a red `??`, so none can slip through.
- `% TODO` comments inside the sources list what the author must fill or confirm.

## Status (2026-11-25)

| Part | State |
|---|---|
| Cover, abstract, acknowledgements, lists | drafted; metadata (full name, supervisor, year) pending |
| Ch. 1 Introduction | drafted |
| Ch. 2 Literature review | drafted; Table 2.1 cells must be verified; FedDAE missing |
| Ch. 3 Methodology | drafted; attack protocol pending |
| Ch. 4 Setup | drafted |
| Ch. 5 Results | 5.1, 5.3, 5.4, 5.7 have audited numbers; cross-module paired tests, privacy measurements, stale-head measurement, round-100 re-scores pending |
| Ch. 6 Discussion, Ch. 7 Conclusion | drafted; revise after pending results and the advisor's decisions |
| Appendices | A complete; B, C partially pending |
