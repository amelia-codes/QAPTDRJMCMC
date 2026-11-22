# The method figures in figures/methods/: pseudocode, a flowchart, the main circuit and its
# subcircuits for every method. Each .tex file is compiled on its own into rendered/<name>.pdf
# beside it, which is what the file falls back to when its packages are not loaded. A test paper
# then \input's every file twice, as an article would: with the packages, and with graphicx alone.
#
#   python3.12 circuit_diagrams.py

import subprocess
import tempfile
from pathlib import Path

import plot_style as style


METHODS = style.FIGURES / "methods"

# Every package a figure can draw with
PACKAGES = r"\usepackage{amsmath,graphicx,tikz,quantikz,algpseudocode}"

# One figure on its own, with its float, caption and label removed
RENDER = r"""\documentclass[varwidth=18cm,border=5pt]{standalone}
%s
\makeatletter
\renewenvironment{figure}[1][]{\def\@captype{figure}}{}
\renewenvironment{figure*}[1][]{\def\@captype{figure}}{}
\renewcommand\caption[2][]{}
\renewcommand\label[1]{}
\makeatother
\begin{document}
\input{%s}
\end{document}
"""

# A two-column paper, like the manuscript
PAPER = r"""\documentclass[twocolumn,aps,prd,floatfix]{revtex4-2}
%s
\begin{document}
%s
\end{document}
"""


# Compile a document in its own folder; any LaTeX error, overfull box or clashing label is a failure
def compile(document):
    result = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-no-shell-escape",
                             document.name], cwd=document.parent, capture_output=True, text=True)
    log = document.with_suffix(".log").read_text(errors="replace")

    for problem in ("Overfull \\hbox", "multiply defined"):
        if problem in log:
            raise RuntimeError(f"{problem} in {document.name}")
    if result.returncode:
        raise RuntimeError(f"could not compile {document.name}:\n{result.stdout[-4000:]}")


def plot():
    figures = sorted(path for path in METHODS.rglob("*.tex") if path.name != "style.tex")
    style_changed = (METHODS / "style.tex").stat().st_mtime

    with tempfile.TemporaryDirectory(prefix="quovar-methods-") as build:
        build = Path(build)

        # Render each figure that changed since its PDF was written
        for figure in figures:
            pdf = figure.parent / "rendered" / f"{figure.stem}.pdf"
            if pdf.exists() and pdf.stat().st_mtime > max(figure.stat().st_mtime, style_changed):
                continue

            document = build / "render.tex"
            document.write_text(RENDER % (PACKAGES, figure.resolve().as_posix()))
            compile(document)
            pdf.parent.mkdir(exist_ok=True)
            pdf.write_bytes(document.with_suffix(".pdf").read_bytes())

        # \input every figure into a paper: drawn natively, then from the rendered PDFs alone
        inputs = "\n".join(rf"\input{{{figure.resolve().as_posix()}}}" + "\n\\clearpage" for figure in figures)
        for preamble in (PACKAGES, r"\usepackage{graphicx}"):
            document = build / "paper.tex"
            document.write_text(PAPER % (preamble, inputs))
            compile(document)


if __name__ == "__main__":
    plot()
