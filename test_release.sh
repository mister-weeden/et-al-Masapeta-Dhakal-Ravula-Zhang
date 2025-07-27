#!/bin/bash

echo "🚀 Testing GitHub Actions Release Workflow"
echo "=========================================="
echo ""

# Function to make a test change
make_test_change() {
    local change_type=$1
    local message=$2
    
    echo "📝 Making test change: $change_type"
    
    # Add a comment to main.tex with timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    sed -i '' "3i\\
% Test change ($change_type): $timestamp" CVPR_1494/main.tex
    
    # Commit and push
    git add CVPR_1494/main.tex
    git commit -m "$message"
    git push origin main
    
    echo "✅ Change committed and pushed"
    echo "🔗 Check workflow: https://github.com/mister-weeden/et-al-Masapeta-Dhakal-Ravula-Zhang/actions"
    echo ""
}

# Show current status
echo "📊 Current Repository Status:"
echo "Latest commit: $(git log --oneline -1)"
echo "Latest tag: $(git describe --tags --abbrev=0 2>/dev/null || echo 'No tags yet')"
echo ""

# Menu for test options
echo "🎯 Choose test type:"
echo "1. Patch release (bug fix)"
echo "2. Minor release (new feature)"
echo "3. Major release (breaking change)"
echo "4. Check workflow status"
echo "5. View releases"
echo "6. Exit"
echo ""

read -p "Enter your choice (1-6): " choice

case $choice in
    1)
        make_test_change "patch" "fix: correct formatting in peer review document

- Minor formatting improvements
- Fix spacing issues in bibliography
- Update document metadata"
        ;;
    2)
        make_test_change "minor" "feat: enhance technical analysis section

- Add detailed implementation suggestions
- Expand computer vision library recommendations
- Include performance benchmarking guidelines"
        ;;
    3)
        make_test_change "major" "breaking: restructure peer review document

- Major reorganization of review sections
- Updated evaluation criteria
- Breaking changes to document structure"
        ;;
    4)
        echo "🔍 Opening workflow status..."
        open "https://github.com/mister-weeden/et-al-Masapeta-Dhakal-Ravula-Zhang/actions"
        ;;
    5)
        echo "📦 Opening releases page..."
        open "https://github.com/mister-weeden/et-al-Masapeta-Dhakal-Ravula-Zhang/releases"
        ;;
    6)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "⏳ Workflow should start in a few seconds..."
echo "📋 What happens next:"
echo "  1. GitHub Actions will detect the change"
echo "  2. LaTeX PDF will be compiled"
echo "  3. Version will be calculated based on commit message"
echo "  4. New release will be created automatically"
echo "  5. PDF will be uploaded to the release"
echo ""
echo "🔗 Monitor progress:"
echo "  - Actions: https://github.com/mister-weeden/et-al-Masapeta-Dhakal-Ravula-Zhang/actions"
echo "  - Releases: https://github.com/mister-weeden/et-al-Masapeta-Dhakal-Ravula-Zhang/releases"
