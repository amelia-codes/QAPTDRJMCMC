# The method figures in figures/methods/: pseudocode, a flowchart, the main circuit and its
# subcircuits for every method. Each .tex file is only the drawing; the paper puts it in a float
# with its own caption and label. Each file is compiled on its own into rendered/<name>.pdf
# beside it, which is what the file falls back to when its packages are not loaded. A test paper
# then \input's every file into a float twice, as the paper would: with the packages, and with
# graphicx alone.
#
# The drawings show classical AMR only when the document defines \qvClassicalAMR. With
# --classical-amr the rendered PDFs and test papers define it; otherwise they show quantum AMR alone.
#
#   python3.12 circuit_diagrams.py                   # quantum AMR only
#   python3.12 circuit_diagrams.py --classical-amr   # with classical and hybrid AMR

import re
import subprocess
import tempfile
from pathlib import Path

import plot_style as style


METHODS = style.FIGURES / "methods"

# Every package a figure can draw with, and whether the drawings show classical AMR
PACKAGES = r"\usepackage{amsmath,graphicx,tikz,quantikz,algpseudocode}"
OPTIONS = r"\def\qvClassicalAMR{}" if style.CLASSICAL_AMR else ""

# The setting the rendered PDFs were last drawn with; a change re-renders them all
STAMP = style.FIGURES / ".method_figures_mode"
MODE = "classical and quantum AMR" if style.CLASSICAL_AMR else "quantum AMR"

# One figure on its own
RENDER = r"""\documentclass[varwidth=18cm,border=5pt]{standalone}
%s
\begin{document}
\input{%s}
\end{document}
"""

# A two-column paper, like the manuscript, and one of its floats: (float, file, number)
PAPER = r"""\documentclass[twocolumn,aps,prd,floatfix]{revtex4-2}
%s
\begin{document}
%s
\end{document}
"""
FLOAT = r"""\begin{%s}
\centering
\input{%s}
\caption{Test figure %d.}
\label{fig:test-%d}
\end{%s}
\clearpage
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
    same_mode = STAMP.exists() and STAMP.read_text() == MODE

    with tempfile.TemporaryDirectory(prefix="quovar-methods-") as build:
        build = Path(build)

        # Render each figure that changed since its PDF was written, or every figure if the
        # classical AMR setting changed
        for figure in figures:
            pdf = figure.parent / "rendered" / f"{figure.stem}.pdf"
            if same_mode and pdf.exists() and pdf.stat().st_mtime > max(figure.stat().st_mtime, style_changed):
                continue

            document = build / "render.tex"
            document.write_text(RENDER % (PACKAGES + OPTIONS, figure.resolve().as_posix()))
            compile(document)
            pdf.parent.mkdir(exist_ok=True)
            pdf.write_bytes(document.with_suffix(".pdf").read_bytes())

        # \input every figure into its suggested float (figure or figure*) in a paper: drawn
        # natively, then from the rendered PDFs alone
        floats = []
        for number, figure in enumerate(figures):
            float_name = re.search(r"inside your own (figure\*?)\.", figure.read_text()).group(1)
            floats.append(FLOAT % (float_name, figure.resolve().as_posix(), number, number, float_name))
        inputs = "".join(floats)
        STAMP.write_text(MODE)
        for preamble in (PACKAGES + OPTIONS, r"\usepackage{graphicx}" + OPTIONS):
            document = build / "paper.tex"
            document.write_text(PAPER % (preamble, inputs))
            compile(document)


if __name__ == "__main__":
    plot()
