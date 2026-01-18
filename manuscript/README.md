# Manuscript

This directory contains the LaTeX source for the ICML 2026 paper "Quantifying the Effect of Test Set Contamination on Generative Evaluations".

## Building the PDF

### Prerequisites

- A LaTeX distribution (e.g., TeX Live, MiKTeX)
- BibTeX for bibliography processing

### Quick Build

From this directory, run:

```bash
pdflatex 00_main.tex
bibtex 00_main
pdflatex 00_main.tex
pdflatex 00_main.tex
```

The multiple `pdflatex` runs are necessary to resolve cross-references and citations.

### One-liner

```bash
pdflatex -interaction=nonstopmode 00_main.tex && bibtex 00_main && pdflatex -interaction=nonstopmode 00_main.tex && pdflatex -interaction=nonstopmode 00_main.tex
```

### Using latexmk (recommended)

If you have `latexmk` installed, you can use:

```bash
latexmk -pdf 00_main.tex
```

To clean build artifacts:

```bash
latexmk -c
```

## File Structure

- `00_main.tex` - Main document file
- `01_introduction.tex` - Introduction section
- `02_methodology.tex` - Methodology section
- `03_pretraining.tex` - Pretraining results
- `04_further_training.tex` - Overtraining and SFT results
- `05_generation.tex` - Temperature and solution length analysis
- `06_discussion.tex` - Discussion and conclusions
- `99_appendix.tex` - Appendix with related work and implementation details
- `references_rylan.bib` - Bibliography
- `math_commands.tex` - Custom math commands
- `figures/` - Figure files (PDF, PNG)
