# GitHub Actions LaTeX PDF Build Workflow

## Overview
This repository includes a GitHub Actions workflow that automatically builds the LaTeX peer review document into a PDF whenever changes are made to the LaTeX source files.

## Workflow Features

### 🚀 Automatic Triggers
- **Push to main branch**: Triggers when `.tex` or `.bib` files in `CVPR_1494/` are modified
- **Pull requests**: Triggers when `.tex` or `.bib` files in `CVPR_1494/` are modified
- **Manual dispatch**: Can be triggered manually from the GitHub Actions interface

### 📦 Workflow Capabilities
- Compiles `CVPR_1494/main.tex` to PDF using LaTeX
- Installs all required LaTeX packages automatically
- Provides detailed error logging and debugging
- Uploads the generated PDF as a downloadable artifact
- Retains build artifacts for 30 days
- Includes debug mode for troubleshooting

## How to Use

### 🔄 Automatic Build
1. Make changes to any `.tex` or `.bib` file in the `CVPR_1494/` directory
2. Commit and push your changes to the main branch
3. The workflow will automatically start building the PDF
4. Check the Actions tab to monitor progress

### 🎯 Manual Trigger
1. Go to the [Actions tab](https://github.com/mister-weeden/et-al-Masapeta-Dhakal-Ravula-Zhang/actions)
2. Click on "Build LaTeX PDF" workflow
3. Click the "Run workflow" button
4. Optionally enable debug mode for detailed output
5. Click "Run workflow" to start the build

### 📥 Download PDF
1. Go to the [Actions tab](https://github.com/mister-weeden/et-al-Masapeta-Dhakal-Ravula-Zhang/actions)
2. Click on a completed workflow run
3. Scroll down to the "Artifacts" section
4. Download the `peer-review-pdf-[run-number]` artifact
5. Extract the ZIP file to get the PDF

## Workflow Configuration

### 📁 File Location
- Workflow file: `.github/workflows/build-pdf-on-pull-request.yaml`
- Main LaTeX file: `CVPR_1494/main.tex`
- Bibliography files: `CVPR_1494/main.bib`, `CVPR_1494/reference.bib`

### 🔧 LaTeX Packages
The workflow automatically installs these LaTeX packages:
- collection-fontsrecommended
- natbib, cleveref, soul, balance
- enumitem, booktabs, caption, subcaption
- url, hyperref, xcolor
- amsmath, amsfonts, amssymb
- graphicx, epsfig, times
- silence, etoolbox

### ⚙️ Build Settings
- Compiler: pdflatex
- Arguments: `-pdf -file-line-error -halt-on-error -interaction=nonstopmode`
- Working directory: Repository root
- Target file: `CVPR_1494/main.tex`

## Troubleshooting

### 🐛 Debug Mode
Enable debug mode when manually triggering the workflow to get:
- Directory structure listing
- File existence verification
- Detailed build logs

### 📋 Build Logs
If the build fails:
1. Check the workflow run details in the Actions tab
2. Look for error messages in the "Set up LaTeX environment" step
3. Download build logs artifact (available for 7 days after failure)

### 🔍 Common Issues
- **Missing packages**: The workflow installs common packages, but rare packages may need to be added
- **File paths**: Ensure all included files use correct relative paths
- **Bibliography errors**: Check that all cited references exist in the .bib files

## Local Testing
Before pushing changes, you can test the build locally:
```bash
cd CVPR_1494
make clean && make paper
# or
./build.sh
```

## Workflow Status
- ✅ Workflow file created and configured
- ✅ Manual trigger capability added
- ✅ Automatic triggers on push/PR configured
- ✅ LaTeX package installation automated
- ✅ PDF artifact upload configured
- ✅ Error handling and logging implemented

## Next Steps
1. Visit the [Actions tab](https://github.com/mister-weeden/et-al-Masapeta-Dhakal-Ravula-Zhang/actions) to verify the workflow is running
2. Test manual trigger functionality
3. Make a small change to a .tex file to test automatic triggering
4. Download and verify the generated PDF artifact
