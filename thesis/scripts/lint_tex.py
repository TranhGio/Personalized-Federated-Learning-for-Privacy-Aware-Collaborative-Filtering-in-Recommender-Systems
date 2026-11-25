#!/usr/bin/env python3
"""Structural checks of the thesis sources that need no TeX installation.

Checks, over every ``.tex`` reachable from ``main.tex`` through ``\\input``:

* braces balance per file (ignoring comments and ``\\{`` / ``\\}``);
* every ``\\begin{env}`` has a matching ``\\end{env}`` in order;
* every ``\\ref``/``\\eqref``/``\\pageref`` target has a ``\\label``;
* every ``\\cite`` key exists in ``references.bib``;
* every ``\\input{...}`` and ``\\includegraphics{...}`` file exists;
* every macro ``\\Name`` used in the prose and defined nowhere in the preamble is
  defined in ``tables/numbers.tex`` (catches a renamed macro before the TeX run);
* a summary of ``\\pending`` and ``\\decision`` placeholders per file.

Exit status 1 if any check fails. Run from the repository root or from ``thesis/``.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

HERE = Path(__file__).resolve().parent
THESIS = HERE.parent
MAIN = THESIS / "main.tex"

# Macros that LaTeX, the packages or fptmse.sty define; anything else that looks like a
# custom macro (capitalised, letters only) must come from tables/numbers.tex.
KNOWN_PREFIXES = ("Ds", "Run", "Ndcg", "Hr", "Pt", "Head", "Params", "Stale", "Cold", "Cache",
                  "Inflation", "Gain", "Delta", "Published", "Official", "Central", "Dim", "Mlp",
                  "N", "Participation", "Seed", "Clients", "Expected", "Gap", "Prototype",
                  "Contrastive", "Alpha")


def strip_comments(text: str) -> str:
    out = []
    for line in text.splitlines():
        buf = []
        i = 0
        while i < len(line):
            ch = line[i]
            if ch == "\\" and i + 1 < len(line):
                buf.append(line[i:i + 2])
                i += 2
                continue
            if ch == "%":
                break
            buf.append(ch)
            i += 1
        out.append("".join(buf))
    return "\n".join(out)


def collect_files(main: Path) -> List[Path]:
    seen: List[Path] = []

    def walk(path: Path) -> None:
        if path in seen:
            return
        seen.append(path)
        text = strip_comments(path.read_text(encoding="utf-8"))
        for m in re.finditer(r"\\input\{([^}]+)\}", text):
            target = THESIS / m.group(1)
            if target.suffix != ".tex":
                target = target.with_suffix(".tex")
            if target.exists():
                walk(target)

    walk(main)
    return seen


def check_braces(path: Path, text: str, errors: List[str]) -> None:
    depth = 0
    for ln, line in enumerate(text.splitlines(), 1):
        i = 0
        while i < len(line):
            if line[i] == "\\" and i + 1 < len(line):
                i += 2
                continue
            if line[i] == "{":
                depth += 1
            elif line[i] == "}":
                depth -= 1
                if depth < 0:
                    errors.append(f"{path.name}:{ln}: unmatched '}}'")
                    depth = 0
            i += 1
    if depth != 0:
        errors.append(f"{path.name}: braces unbalanced at end of file (depth {depth})")


def check_envs(path: Path, text: str, errors: List[str]) -> None:
    stack: List[Tuple[str, int]] = []
    for ln, line in enumerate(text.splitlines(), 1):
        for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", line):
            kind, env = m.group(1), m.group(2)
            if kind == "begin":
                stack.append((env, ln))
            else:
                if not stack or stack[-1][0] != env:
                    if stack:
                        expected = "begin{%s} at line %d" % (stack[-1][0], stack[-1][1])
                        stack.pop()
                    else:
                        expected = "anything"
                    errors.append("%s:%d: end{%s} does not match %s" % (path.name, ln, env, expected))
                else:
                    stack.pop()
    for env, ln in stack:
        errors.append(f"{path.name}:{ln}: \\begin{{{env}}} never closed")


def main() -> int:
    errors: List[str] = []
    files = collect_files(MAIN)
    texts: Dict[Path, str] = {p: strip_comments(p.read_text(encoding="utf-8")) for p in files}

    labels: Set[str] = set()
    refs: List[Tuple[Path, str]] = []
    cites: List[Tuple[Path, str]] = []
    used_macros: Set[str] = set()
    placeholders: Dict[str, Tuple[int, int]] = {}

    for p, t in texts.items():
        check_braces(p, t, errors)
        check_envs(p, t, errors)
        labels.update(re.findall(r"\\label\{([^}]+)\}", t))
        refs += [(p, r) for group in re.findall(r"\\(?:ref|eqref|pageref|autoref)\{([^}]+)\}", t)
                 for r in group.split(",")]
        cites += [(p, c.strip()) for group in re.findall(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]+)\}", t)
                  for c in group.split(",")]
        for m in re.finditer(r"\\input\{([^}]+)\}", t):
            target = THESIS / m.group(1)
            if target.suffix != ".tex":
                target = target.with_suffix(".tex")
            if not target.exists():
                errors.append(f"{p.name}: \\input target missing: {m.group(1)}")
        for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", t):
            if not (THESIS / m.group(1)).exists():
                errors.append(f"{p.name}: figure missing: {m.group(1)}")
        if p.name not in ("numbers.tex", "fptmse.sty"):
            for m in re.finditer(r"\\([A-Z][A-Za-z]*)([0-9]?)", t):
                name, digit = m.group(1), m.group(2)
                if digit:
                    errors.append(f"{p.name}: macro \\{name}{digit}... contains a digit (invalid LaTeX name)")
                if name.startswith(KNOWN_PREFIXES):
                    used_macros.add(name)
        placeholders[p.name] = (len(re.findall(r"\\pending\{", t)), len(re.findall(r"\\decision\{", t)))

    for p, r in refs:
        if r not in labels:
            errors.append(f"{p.name}: \\ref to unknown label '{r}'")

    bib = (THESIS / "references.bib").read_text(encoding="utf-8")
    bibkeys = set(re.findall(r"^@\w+\{([^,]+),", bib, flags=re.M))
    for p, c in cites:
        if c not in bibkeys:
            errors.append(f"{p.name}: \\cite to unknown bib key '{c}'")

    numbers = (THESIS / "tables" / "numbers.tex").read_text(encoding="utf-8")
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", numbers))
    sty = (THESIS / "fptmse.sty").read_text(encoding="utf-8")
    defined |= set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", sty))
    defined |= set(re.findall(r"\\newcommand\*?\{\\([A-Za-z]+)\}", strip_comments(MAIN.read_text(encoding="utf-8"))))
    latex_builtin = {"NeedsTeXFormat", "ProvidesPackage", "RequirePackage", "DeclareMathOperator",
                     "MakeUppercase", "Large", "LARGE", "Huge", "Stealth", "Delta"}
    for name in sorted(used_macros):
        if name not in defined and name not in latex_builtin:
            errors.append(f"macro \\{name} used but not defined in tables/numbers.tex")

    print(f"files checked: {len(files)}; labels: {len(labels)}; refs: {len(refs)}; cites: {len(cites)}; "
          f"number macros used: {len(used_macros)}")
    tot_p = sum(v[0] for v in placeholders.values())
    tot_d = sum(v[1] for v in placeholders.values())
    print(f"placeholders: {tot_p} \\pending, {tot_d} \\decision")
    for name, (np_, nd_) in placeholders.items():
        if np_ or nd_:
            print(f"  {name}: {np_} pending, {nd_} decision")
    if errors:
        print("\nERRORS:")
        for e in errors:
            print("  " + e)
        return 1
    print("\nOK: no structural errors found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
