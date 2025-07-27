# CVPR 2025 Peer Review: "Show and Segment"

Welcome! This repo contains the LaTeX structure and resources for your team’s peer review of CVPR 2025 Paper #1494:  
**"Show and Segment: Universal Medical Image Segmentation via In-Context Learning"**

---

## Quick Start

1. **Pick Your Section**  
   - See [Team Assignments](#team-assignments) below.
   - Coordinate with Akhila, Zezheng, and Sriya to select sections 1–4.

2. **Edit Your Section**  
   - Replace placeholder text in your `.tex` file (see [File Structure](#file-structure)).
   - Use Overleaf or compile locally (see [Compile Instructions](#compile-instructions)).

3. **Submit & Review**  
   - Commit changes, sync, and create a pull request.
   - Review the full document before final submission.

---

## File Structure

```
....
CVPR_1494/
├── main.tex                  # Main document
├── preamble.tex              # Packages & commands
├── main.bib                  # Bibliography (datasets)
├── 0_executive_summary.tex   # Phaninder (Lead)
├── 1_technical_contribution.tex
├── 2_experimental_methodology.tex
├── 3_results_validation.tex
├── 4_literature_review.tex
├── 5_presentation_clarity.tex # Scott
├── 6_detailed_comments.tex   # All
├── 7_questions_authors.tex   # All
├── 8_minor_issues.tex        # Scott
└── 9_meta_review.tex         # Phaninder (Lead)
....
GUIDELINES/   # IEEE GUIDELINES (DO NOT CHANGE)
....
IEEE_1494/    # Show and Segment: PUBLICATION TO REVIEW
....
Docs.md...
```

---

## Team Assignments

- **Phaninder Reddy Masapeta (Lead):** Executive Summary (0), Meta-Review (9)
- **Scott Weeden:** Presentation & Clarity (5), Minor Issues (8)
- **Akhila Ravula, Zezheng Zhang, Sriya Dhakal:**  
  - Choose from:  
    - Technical Contribution (1)  
    - Experimental Methodology (2)  
    - Results Validation (3)  
    - Literature Review (4)
- **All Members:** Detailed Comments (6), Questions for Authors (7)

---

## Compile Instructions

- **Overleaf:** Upload all files and compile.
- **Local:**  
  ```bash
  pdflatex main.tex
  bibtex main
  pdflatex main.tex
  pdflatex main.tex
  ```
  Or simply:
  ```bash
  make
  ```

---

## Tips

- Replace all placeholder text with your review.
- Use color-coded annotations:  
  - Green = Strengths  
  - Red = Weaknesses  
  - Blue = Suggestions
- Keep terminology and assessment consistent.
- The Team Lead will do the final integration.

---

**Helpful Links:**  
- [Team Assignment Guide](./Team_Assignment_Guide.md)  
- [Dataset Summary](./Dataset_Summary.md)  
- [Submit Your Section](./Submit_Findings.md)  
- [Commands Quick Review](./review_commands_reference.md)

---

*Let’s make this review clear, constructive, and collaborative!*