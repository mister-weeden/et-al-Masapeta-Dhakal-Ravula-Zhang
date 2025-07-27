# GitHub Actions LaTeX PDF Build Workflow with Automated Releases

## Overview
This repository includes a comprehensive GitHub Actions workflow that automatically builds the LaTeX peer review document into a PDF and creates versioned releases whenever changes are made to the LaTeX source files.

## 🚀 Workflow Features

### 🔄 Automatic Triggers
- **Push to main branch**: Triggers when `.tex` or `.bib` files in `CVPR_1494/` are modified
- **Pull requests**: Triggers when `.tex` or `.bib` files in `CVPR_1494/` are modified
- **Manual dispatch**: Can be triggered manually from the GitHub Actions interface

### 📦 Release Management
- **Automatic versioning**: Uses semantic versioning (major.minor.patch)
- **Smart version detection**: Analyzes commit messages for version hints
- **Release creation**: Automatically creates GitHub releases on successful builds
- **Change summaries**: Generates categorized change logs for each release
- **Asset uploads**: Includes both `main.pdf` and versioned PDF files

### 🏷️ Version Detection Logic
The workflow automatically determines the version bump type:
- **Major**: Triggered by commits containing "breaking" or "major"
- **Minor**: Triggered by commits containing "feat", "feature", or "minor"
- **Patch**: Default for all other changes
- **Manual override**: Can be specified when manually triggering the workflow

### 📋 Release Content
Each release includes:
- **Versioned PDF files**: Both `main.pdf` and `peer-review-vX.Y.Z.pdf`
- **Categorized changelog**: Features, bug fixes, documentation, and other changes
- **Build information**: Date, commit SHA, and workflow run details
- **Automatic tagging**: Git tags for version tracking

## 📊 Status Badges

The repository README includes several status badges:

- **Build Status**: Shows if the latest build passed or failed
- **Latest Release**: Displays the current release version
- **PDF Download**: Direct link to download the latest PDF
- **License**: Academic license indicator

## How to Use

### 🔄 Automatic Build and Release
1. Make changes to any `.tex` or `.bib` file in the `CVPR_1494/` directory
2. Commit with descriptive messages (use "feat:" for features, "fix:" for bugs)
3. Push your changes to the main branch
4. The workflow will automatically:
   - Build the PDF
   - Determine the appropriate version bump
   - Create a new release with the PDF
   - Update status badges

### 🎯 Manual Trigger with Version Control
1. Go to the [Actions tab](https://github.com/mister-weeden/et-al-Masapeta-Dhakal-Ravula-Zhang/actions)
2. Click on "Build LaTeX PDF" workflow
3. Click the "Run workflow" button
4. Choose options:
   - **Debug mode**: Enable for detailed output
   - **Release type**: Select patch/minor/major
5. Click "Run workflow" to start the build

### 📥 Download PDF
Multiple ways to access the PDF:

**From Releases (Recommended):**
1. Go to the [Releases page](https://github.com/mister-weeden/et-al-Masapeta-Dhakal-Ravula-Zhang/releases)
2. Click on the latest release
3. Download either `main.pdf` or the versioned PDF

**Direct Download:**
- Click the "PDF Download" badge in the README
- Or use the direct link: [Latest PDF](https://github.com/mister-weeden/et-al-Masapeta-Dhakal-Ravula-Zhang/releases/latest/download/main.pdf)

**From Artifacts:**
1. Go to the [Actions tab](https://github.com/mister-weeden/et-al-Masapeta-Dhakal-Ravula-Zhang/actions)
2. Click on a completed workflow run
3. Download from the "Artifacts" section

## 🔧 Workflow Configuration

### 📁 File Locations
- **Workflow file**: `.github/workflows/build-pdf-on-pull-request.yaml`
- **Main LaTeX file**: `CVPR_1494/main.tex`
- **Bibliography files**: `CVPR_1494/main.bib`, `CVPR_1494/reference.bib`
- **Status badges**: `README.md`

### 🏗️ Build Process
1. **Checkout**: Fetches full repository history for version calculation
2. **Version calculation**: Determines next version based on commits and manual input
3. **Change summary**: Generates categorized changelog from recent commits
4. **LaTeX compilation**: Builds PDF with all required packages
5. **Verification**: Confirms PDF creation and creates versioned copy
6. **Artifact upload**: Stores PDF files as downloadable artifacts
7. **Release creation**: Creates GitHub release with PDFs and changelog

### 📦 LaTeX Environment
The workflow automatically installs these packages:
- **Core**: collection-fontsrecommended, natbib, cleveref
- **Formatting**: soul, balance, enumitem, booktabs
- **Graphics**: caption, subcaption, url, hyperref, xcolor
- **Math**: amsmath, amsfonts, amssymb
- **Images**: graphicx, epsfig
- **Fonts**: times
- **Utilities**: silence, etoolbox

### ⚙️ Build Settings
- **Compiler**: pdflatex
- **Arguments**: `-pdf -file-line-error -halt-on-error -interaction=nonstopmode`
- **Working directory**: Repository root
- **Target file**: `CVPR_1494/main.tex`
- **Artifact retention**: 90 days for PDFs, 7 days for build logs

## 🐛 Troubleshooting

### 🔍 Debug Mode
Enable debug mode when manually triggering to get:
- Directory structure listing
- File existence verification
- Detailed build logs
- Version calculation details

### 📋 Build Failures
If the build fails:
1. Check the workflow run details in the Actions tab
2. Look for error messages in the "Set up LaTeX environment" step
3. Download build logs artifact (available for 7 days)
4. Check for missing packages or file path issues

### 🏷️ Version Issues
If versions aren't incrementing correctly:
- Check that commits contain appropriate keywords
- Verify the latest tag exists and follows semantic versioning
- Use manual trigger with explicit version type
- Check the "Calculate version" step output

### 📊 Badge Issues
If badges aren't updating:
- Badges may take a few minutes to refresh
- Check that the workflow file path in badges matches actual location
- Verify repository name in badge URLs is correct

## 🔄 Release Workflow

### 📈 Version Progression
- **v0.0.1**: Initial release
- **v0.0.2**: Patch updates (bug fixes, minor changes)
- **v0.1.0**: Minor updates (new features, enhancements)
- **v1.0.0**: Major updates (breaking changes, major milestones)

### 📝 Commit Message Guidelines
For automatic version detection:
- `feat: add new analysis section` → Minor version bump
- `fix: correct citation formatting` → Patch version bump
- `breaking: restructure document layout` → Major version bump
- `docs: update README` → Patch version bump

### 🏷️ Release Naming
- **Tag**: `vX.Y.Z` (e.g., `v1.2.3`)
- **Title**: `Peer Review vX.Y.Z`
- **Assets**: `main.pdf` and `peer-review-vX.Y.Z.pdf`

## 📊 Monitoring

### 🎯 Status Checks
- **Build Status Badge**: Shows current build status
- **Latest Release Badge**: Displays current version
- **Actions Tab**: Detailed workflow run history
- **Releases Page**: Complete release history with changelogs

### 📈 Metrics
- **Build time**: Typically 3-5 minutes
- **Success rate**: Monitored via badge status
- **Artifact size**: PDF files typically 150-200KB
- **Release frequency**: Automatic on every successful main branch build

## 🚀 Advanced Features

### 🔄 Conditional Release Creation
- Releases are only created for pushes to main branch
- Pull requests build PDFs but don't create releases
- Manual triggers can create releases regardless of branch

### 📦 Multi-Asset Releases
Each release includes:
- `main.pdf`: Standard filename for direct linking
- `peer-review-vX.Y.Z.pdf`: Versioned filename for archival
- Detailed changelog with categorized changes
- Build metadata and links

### 🏷️ Smart Tagging
- Automatic Git tag creation
- Semantic version compliance
- Tag-based version calculation for subsequent builds
- Protection against duplicate tags

## 📞 Support

For issues with the workflow:
1. Check the [Actions tab](https://github.com/mister-weeden/et-al-Masapeta-Dhakal-Ravula-Zhang/actions) for recent runs
2. Review build logs and error messages
3. Create an issue in the repository
4. Contact the team lead for assistance

---

**Workflow Status**: ✅ Fully operational with automated releases  
**Last Updated**: Auto-updated on every successful build  
**Next Enhancement**: Consider adding PDF diff generation between versions
