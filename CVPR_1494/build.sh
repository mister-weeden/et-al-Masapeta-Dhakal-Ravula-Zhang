#!/bin/bash

set -e

MAIN_TEX=cvpr
REBUTTAL_TEX=CvprRebuttal

echo "🧱 Building main paper ($MAIN_TEX.tex)..."
pdflatex $MAIN_TEX.tex
bibtex $MAIN_TEX
pdflatex $MAIN_TEX.tex
pdflatex $MAIN_TEX.tex
echo "✅ Paper compiled: $MAIN_TEX.pdf"

if [ -f "$REBUTTAL_TEX.tex" ]; then
    echo "🧾 Building rebuttal ($REBUTTAL_TEX.tex)..."
    pdflatex $REBUTTAL_TEX.tex
    echo "✅ Rebuttal compiled: $REBUTTAL_TEX.pdf"
fi
