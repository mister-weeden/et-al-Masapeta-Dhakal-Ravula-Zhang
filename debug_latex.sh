#!/bin/bash

echo "🔍 LaTeX Build Troubleshooting Script"
echo "====================================="
echo ""

cd CVPR_1494

echo "📋 Checking LaTeX installation..."
which pdflatex || echo "❌ pdflatex not found"
which bibtex || echo "❌ bibtex not found"
echo ""

echo "📦 Checking required packages..."
packages=(
    "natbib"
    "cleveref" 
    "soul"
    "balance"
    "enumitem"
    "booktabs"
    "caption"
    "subcaption"
    "url"
    "hyperref"
    "xcolor"
    "amsmath"
    "amsfonts"
    "amssymb"
    "graphicx"
    "epsfig"
    "times"
    "silence"
    "etoolbox"
)

for package in "${packages[@]}"; do
    if kpsewhich ${package}.sty >/dev/null 2>&1; then
        echo "✅ $package.sty found"
    else
        echo "❌ $package.sty NOT found"
    fi
done

echo ""
echo "📄 Checking main document structure..."
if [ -f "main.tex" ]; then
    echo "✅ main.tex exists"
    echo "📝 Document class and packages used:"
    grep -E "\\\\documentclass|\\\\usepackage" main.tex | head -10
else
    echo "❌ main.tex not found"
fi

echo ""
echo "📚 Checking bibliography files..."
for bibfile in main.bib reference.bib; do
    if [ -f "$bibfile" ]; then
        echo "✅ $bibfile exists ($(wc -l < $bibfile) lines)"
    else
        echo "❌ $bibfile not found"
    fi
done

echo ""
echo "🔧 Attempting local build..."
echo "Running: pdflatex main.tex"
if pdflatex -interaction=nonstopmode main.tex > build.log 2>&1; then
    echo "✅ First pdflatex pass completed"
else
    echo "❌ First pdflatex pass failed"
    echo "Last 10 lines of error log:"
    tail -10 build.log
    echo ""
    echo "Full log saved to build.log"
fi

if [ -f "main.aux" ]; then
    echo "Running: bibtex main"
    if bibtex main > bibtex.log 2>&1; then
        echo "✅ bibtex completed"
    else
        echo "❌ bibtex failed"
        echo "bibtex errors:"
        cat bibtex.log
    fi
fi

echo ""
echo "🎯 Recommendations:"
echo "1. If packages are missing, install them with: tlmgr install <package>"
echo "2. Check the build.log file for detailed error messages"
echo "3. Ensure all \\cite{} references exist in .bib files"
echo "4. For GitHub Actions, the workflow will handle package installation"
echo ""

if [ -f "main.pdf" ]; then
    echo "✅ PDF successfully created: main.pdf"
    ls -la main.pdf
else
    echo "❌ PDF not created - check errors above"
fi
