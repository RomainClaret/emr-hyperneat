"""Text style guard: the writing patterns removed in 0.2.0 stay out of the prose.

Release 0.2.0 removed Unicode arrows, em-dashes, spaced double hyphens used as dashes, en-dashes
outside numeric ranges, box-drawing characters, check marks and emoji, decorative rule comments,
shouted markers and a few stock marker phrases from the comments, docstrings and documentation.
This test keeps them out. It reads comments and docstrings, never other string literals
(displayed text is reviewed by hand), and the prose of README.md, CHANGELOG.md and docs/*.md
outside code fences, inline code and link targets. Besides pytest it needs only the standard
library.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parent

# Arrows, em-dash, box drawing, check marks and the other dingbats, emoji, written as escapes so
# that this file carries none of them.
BANNED = re.compile("[\u2190-\u21ff\u2014\u2500-\u257f\u2600-\u27bf\U0001f300-\U0001faff]")
EN_DASH = "\u2013"
MARKER = re.compile(r"^\s*(?:CRITICAL|IMPORTANT|NOTE|NB|N\.B\.|NEW|PLUS NEW|FIX|FIXED|CRITICAL FIX|"
                    r"KEY INSIGHT|KEY INNOVATION|KEY POINT|KEY CHANGE|CRITICAL DEBUG|OPTIMIZATION|"
                    r"OPTIMIZED|RESTORED)\s*:")
# "(NEW_" starts an identifier, so "(NEW" must end at a word boundary.
PHRASE = re.compile(r"\(NEW\b[^)\n]*\)|\((?i:key) (?i:innovation|synergy)\)|\bNOVEL:"
                    r"|\b[Cc]ritical innovation:|\b(?i:key innovations?):|\b(?i:key insights?)\b"
                    r"|(?<=\s)-\s+(?i:key (?:innovation|synergy))\b")
RULE_CHARS = set("=-*#~_ ")
TITLED = re.compile(r"^(?P<left>[=\-*#~_]{2,})\s+\S.*?(?:\s(?P<right>[=\-*#~_]{2,}))?$"
                    r"|^[^\s=\-*#~_].*?\s[=\-*#~_]{3,}$")
DASH = re.compile(r"(?<=[^\s-]) --(?: (?=[^\s-])|\s*$)")
COMMAND = re.compile(r"python|pytest|pip |\.py\b|\$ ")
# Cheap tests on the raw file text, each a superset of a rule above; only files that pass one are
# tokenized and parsed.
CANDIDATES = ("(NEW", EN_DASH, " -- ", " --\n", " --\t")
LOWER_CANDIDATES = ("(key innovation", "(key synergy", "novel:", "critical innovation:",
                    "key innovation:", "key innovations:", "key insight", "- key innovation",
                    "- key synergy")
CANDIDATE_RES = (BANNED,
                 re.compile(r"(?:CRITICAL|IMPORTANT|NOTE|NB|N\.B\.|NEW|FIX|FIXED|INSIGHT|"
                            r"INNOVATION|POINT|CHANGE|DEBUG|OPTIMIZATION|OPTIMIZED|RESTORED)\s*:"),
                 re.compile(r"#[ \t]*[=\-*#~_]{2}"),
                 re.compile(r"[^\s=\-*#~_][ \t]+[=\-*#~_]{3,}[ \t]*$", re.M))


def candidate(text: str) -> bool:
    """Whether a file can contain a problem at all."""
    if any(c in text for c in CANDIDATES) or any(r.search(text) for r in CANDIDATE_RES):
        return True
    lower = text.lower()
    return any(c in lower for c in LOWER_CANDIDATES)


def problems(text: str, comment: bool = False, own_line: bool = False) -> list[str]:
    """The banned patterns in one comment body, docstring line or line of Markdown prose."""
    found = [f"character U+{ord(m.group(0)[0]):04X}" for m in BANNED.finditer(text)]
    for m in re.finditer(EN_DASH, text):
        before, after = text[:m.start()].rstrip(" "), text[m.end():].lstrip(" ")
        if not (before[-1:].isdigit() and after[:1].isdigit()):
            found.append("en-dash outside a numeric range")
    if MARKER.match(text):
        found.append("shouted marker")
    found += [f"marker phrase {m.group(0)!r}" for m in PHRASE.finditer(text)]
    if DASH.search(text) and not COMMAND.search(text):
        found.append("ASCII dash")
    if comment and own_line:
        body = text.strip()
        titled = TITLED.match(body)
        dash_prefix = titled and titled.group("left") and not titled.group("right") and len(
            titled.group("left")) < 3  # "-- text" is a dash, not a rule
        if len(body) >= 8 and set(body) <= RULE_CHARS:
            found.append("rule comment")
        elif titled and re.search("[A-Za-z0-9]", body) and not dash_prefix:
            found.append("titled rule")
    return found


def python_problems(path: Path, root: Path) -> list[str]:
    """Problems in the comments and docstrings of one Python file."""
    text = path.read_text(encoding="utf-8")
    if not candidate(text):
        return []
    name = path.relative_to(root)
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type != tokenize.COMMENT or (tok.start[0] == 1 and tok.string.startswith("#!")):
            continue
        own_line = not tok.line[:tok.start[1]].strip()
        body = tok.string[2:] if tok.string.startswith("#:") else tok.string[1:]
        out += [f"{name}:{tok.start[0]}: {p}" for p in problems(body, True, own_line)]
    for node in ast.walk(ast.parse(text)):
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            continue
        in_example = False
        for i, line in enumerate(node.value.value.split("\n")):
            in_example = (in_example and bool(line.strip())) or line.strip().startswith(">>>")
            if not in_example:  # doctest examples and their output are code
                out += [f"{name}:{node.lineno + i}: {p}" for p in problems(line)]
    return out


def markdown_problems(path: Path, root: Path) -> list[str]:
    """Problems in the prose of one Markdown file."""
    out, fence = [], ""
    for n, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
        m = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence:
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence):
                fence = ""
            continue
        if m:
            fence = m.group(1)
            continue
        prose = re.sub(r"(`+).+?\1", "", line)
        prose = re.sub(r"\]\([^)]*\)|<!--.*?-->", "", prose)
        out += [f"{path.relative_to(root)}:{n}: {p}" for p in problems(prose)]
    return out


def python_files(root: Path) -> list[Path]:
    package = root / "emr_hyperneat" if (root / "emr_hyperneat").is_dir() else PACKAGE
    dirs = [d for d in (package, root / "scripts", root / "papers") if d.is_dir()]
    return [p for d in dirs for p in sorted(d.rglob("*.py"))]


def test_comments_and_docstrings_are_clean():
    found = [p for path in python_files(REPO) for p in python_problems(path, REPO)]
    assert not found, f"{len(found)} problems:\n" + "\n".join(found[:60])


@pytest.mark.skipif(not (REPO / "README.md").is_file(), reason="needs the repository checkout")
def test_documentation_prose_is_clean():
    docs = [REPO / "README.md", REPO / "CHANGELOG.md", *sorted((REPO / "docs").glob("*.md"))]
    found = [p for path in docs if path.is_file() for p in markdown_problems(path, REPO)]
    assert not found, f"{len(found)} problems:\n" + "\n".join(found[:60])
