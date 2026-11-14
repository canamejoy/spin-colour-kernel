"""Build the PDF user manuals of ready_for_run/ from their Markdown sources, and copy the notebook there.

    python notebooks/build_manuals.py            # needs mistune and the tectonic LaTeX engine

The Markdown files are the single source; each is converted to LaTeX (headings, paragraphs, lists,
tables, code blocks, inline code, bold, links) and compiled with tectonic.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import mistune

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ready_for_run"
MANUALS = ("MANUAL_en", "MANUAL_es")

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=2.2cm]{geometry}
\usepackage{fontspec}
\setmainfont{DejaVu Sans}
\setmonofont{DejaVu Sans Mono}[Scale=0.88]
\usepackage{xcolor,tabularx,booktabs,enumitem,fvextra,titlesec}
\usepackage[hidelinks]{hyperref}
\definecolor{codebg}{gray}{0.95}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.55em}
\setlist{itemsep=0.2em,topsep=0.3em}
\titleformat{\section}{\large\bfseries}{}{0pt}{}
\titleformat{\subsection}{\normalsize\bfseries}{}{0pt}{}
\renewcommand{\arraystretch}{1.25}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\DefineVerbatimEnvironment{code}{Verbatim}{breaklines,breakanywhere,fontsize=\small,frame=single,
  rulecolor=\color{gray!40},framesep=6pt}
\newcommand{\inl}[1]{\texttt{#1}}
\begin{document}
"""

_ESC = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "%": r"\%", "#": r"\#",
        "_": r"\_", "^": r"\^{}", "~": r"\textasciitilde{}", "⟨": r"\ensuremath{\langle}",
        "⟩": r"\ensuremath{\rangle}", "<": r"\textless{}", ">": r"\textgreater{}"}


def esc(s: str) -> str:
    return "".join(_ESC.get(c, c) for c in s)


def inline(tokens) -> str:
    out = []
    for t in tokens:
        k = t["type"]
        if k == "text":
            out.append(esc(t["raw"]))
        elif k == "codespan":
            out.append(r"\inl{" + esc(t["raw"]) + "}")
        elif k == "strong":
            out.append(r"\textbf{" + inline(t["children"]) + "}")
        elif k == "emphasis":
            out.append(r"\emph{" + inline(t["children"]) + "}")
        elif k == "link":
            url = t["attrs"]["url"]
            text = inline(t["children"])
            out.append(r"\url{" + url + "}" if text == esc(url) else r"\href{" + url + "}{" + text + "}")
        elif k in ("softbreak", "linebreak"):
            out.append(" " if k == "softbreak" else r"\\")
        else:
            raise ValueError(f"unsupported inline token {k}")
    return "".join(out)


def block(t, first_heading: list) -> str:
    k = t["type"]
    if k == "blank_line":
        return ""
    if k == "heading":
        text = inline(t["children"])
        lvl = t["attrs"]["level"]
        if lvl == 1:
            first_heading.append(text)
            return r"{\LARGE\bfseries " + text + r"\par}\vspace{0.6em}" + "\n"
        return (r"\section*{" if lvl == 2 else r"\subsection*{") + text + "}\n"
    if k in ("paragraph", "block_text"):
        return inline(t["children"]) + "\n\n"
    if k == "list":
        env = "enumerate" if t["attrs"].get("ordered") else "itemize"
        items = "".join(r"\item " + "".join(block(c, first_heading) for c in it["children"]).strip() + "\n"
                        for it in t["children"])
        return f"\\begin{{{env}}}\n{items}\\end{{{env}}}\n"
    if k == "block_code":
        return "\\begin{code}\n" + t["raw"].rstrip("\n") + "\n\\end{code}\n"
    if k == "table":
        head = next(c for c in t["children"] if c["type"] == "table_head")
        body = next(c for c in t["children"] if c["type"] == "table_body")
        n = len(head["children"])
        row = lambda cells: " & ".join(inline(c["children"]) for c in cells) + r" \\"
        rows = "\n".join(row(r["children"]) for r in body["children"])
        return (r"{\small\begin{tabularx}{\linewidth}{" + "Y" * n + "}\\toprule\n"
                + row([{"children": [{"type": "strong", "children": c["children"]}]} for c in head["children"]])
                + "\n\\midrule\n" + rows + "\n\\bottomrule\\end{tabularx}}\n\n")
    if k == "thematic_break":
        return r"\noindent\rule{\linewidth}{0.4pt}" + "\n"
    raise ValueError(f"unsupported block token {k}")


def to_latex(md_text: str) -> str:
    tokens = mistune.create_markdown(renderer=None, plugins=["table"])(md_text)
    titles: list = []
    body = "".join(block(t, titles) for t in tokens)
    return PREAMBLE + body + "\\end{document}\n"


def build_pdf(name: str) -> Path:
    tex = to_latex((OUT / f"{name}.md").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / f"{name}.tex"
        src.write_text(tex, encoding="utf-8")
        r = subprocess.run(["tectonic", "-X", "compile", str(src)], capture_output=True, text=True)
        if r.returncode != 0:
            sys.stderr.write(r.stderr[-3000:])
            raise SystemExit(f"{name}: LaTeX compilation failed")
        shutil.copy(Path(d) / f"{name}.pdf", OUT / f"{name}.pdf")
    return OUT / f"{name}.pdf"


if __name__ == "__main__":
    shutil.copy(ROOT / "notebooks" / "spin_simulation.ipynb", OUT / "spin_simulation.ipynb")
    for m in MANUALS:
        print("wrote", build_pdf(m))
