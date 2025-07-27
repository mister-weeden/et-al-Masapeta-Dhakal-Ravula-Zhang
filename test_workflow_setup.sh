#!/bin/bash

echo "🧪 Testing Workflow Setup Locally"
echo "================================="
echo ""

# Test 1: Check if we're in the right directory
echo "📁 Current directory: $(pwd)"
echo ""

# Test 2: Check CVPR_1494 directory structure
echo "📋 CVPR_1494 directory contents:"
if [ -d "CVPR_1494" ]; then
    ls -la CVPR_1494/
    echo ""
    
    # Test 3: Check required files
    echo "🔍 Required files check:"
    files=("main.tex" "cvpr.sty" "main.bib" "balance.sty")
    for file in "${files[@]}"; do
        if [ -f "CVPR_1494/$file" ]; then
            echo "✅ $file exists"
        else
            echo "❌ $file missing"
        fi
    done
    echo ""
    
    # Test 4: Try to compile from CVPR_1494 directory
    echo "🔨 Testing compilation from CVPR_1494 directory:"
    cd CVPR_1494
    echo "Working directory: $(pwd)"
    echo "Files in current directory:"
    ls -la *.tex *.sty *.bib 2>/dev/null || echo "Some files missing"
    echo ""
    
    # Test 5: Check if main.tex can find cvpr.sty
    echo "📄 Checking main.tex for cvpr.sty usage:"
    grep "usepackage{cvpr}" main.tex && echo "✅ cvpr.sty is used in main.tex"
    echo ""
    
    # Test 6: Simulate workflow compilation
    echo "⚙️  Simulating workflow compilation:"
    echo "Command that workflow will run: pdflatex -pdf -file-line-error -halt-on-error -interaction=nonstopmode main.tex"
    
    if command -v pdflatex >/dev/null 2>&1; then
        echo "pdflatex is available, testing..."
        if pdflatex -interaction=nonstopmode main.tex > test_compile.log 2>&1; then
            echo "✅ First pass compilation successful"
            if [ -f "main.pdf" ]; then
                echo "✅ PDF created successfully"
                ls -la main.pdf
            else
                echo "❌ PDF not created"
            fi
        else
            echo "❌ Compilation failed"
            echo "Last 10 lines of error log:"
            tail -10 test_compile.log
        fi
    else
        echo "⚠️  pdflatex not available locally, skipping compilation test"
    fi
    
else
    echo "❌ CVPR_1494 directory not found"
fi

echo ""
echo "🎯 Workflow Configuration Check:"
echo "The GitHub Actions workflow should:"
echo "1. Set working_directory to: CVPR_1494"
echo "2. Set root_file to: main.tex (not CVPR_1494/main.tex)"
echo "3. Ensure cvpr.sty is in the same directory as main.tex"
echo ""
echo "Current workflow settings should be:"
echo "  working_directory: CVPR_1494"
echo "  root_file: main.tex"
