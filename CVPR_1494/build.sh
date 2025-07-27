#!/bin/bash

set -e

# Clean all auxiliary and output files
echo "🧹 Cleaning auxiliary and output files..."
find . -type f \( \
    -name "*.aux" -o -name "*.log" -o -name "*.bbl" -o -name "*.blg" \
    -o -name "*.out" -o -name "*.toc" -o -name "*.lof" -o -name "*.lot" \
    -o -name "*.gz" -o -name "*.nav" -o -name "*.snm" -o -name "*.fdb_latexmk" \
    -o -name "*.fls" -o -name "*.synctex.gz" -o -name "*.pdf" \
    \) -delete

MAIN_TEX=main
REBUTTAL_TEX=CvprRebuttal

# Check for required files
for f in *.cls *.sty *.bst *.bib; do
    if ! ls $f 1> /dev/null 2>&1; then
        echo "⚠️  Warning: Required file type '$f' not found in directory."
    fi
done

echo "🧱 Building main paper ($MAIN_TEX.tex)..."
pdflatex $MAIN_TEX.tex
bibtex $MAIN_TEX
pdflatex $MAIN_TEX.tex
pdflatex $MAIN_TEX.tex
echo "✅ Paper compiled: $MAIN_TEX.pdf"

if [ -f "$REBUTTAL_TEX.tex" ]; then
    echo "🧾 Building rebuttal ($REBUTTAL_TEX.tex)..."
    pdflatex $REBUTTAL_TEX.tex
    bibtex $REBUTTAL_TEX || true
    pdflatex $REBUTTAL_TEX.tex
    pdflatex $REBUTTAL_TEX.tex
    echo "✅ Rebuttal compiled: $REBUTTAL_TEX.pdf"
fi