#!/bin/bash
#!/bin/bash

# Find all .tex files and replace 'section{' with 'section*{'
find . -type f -name "*.tex" -exec sed -i '' 's/\\section{/\\section*{/g' {} +

rm -f *.aux *.log *.bbl *.blg *.out *.toc *.lof *.lot *.gz *.nav *.snm *.fdb_latexmk *.fls

set -e

MAIN_TEX=main
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
