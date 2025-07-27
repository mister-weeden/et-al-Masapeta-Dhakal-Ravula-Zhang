# Makefile for CVPR LaTeX Submission

MAIN_TEX=main
REBUTTAL_TEX=CvprRebuttal
SECTIONS=$(wildcard sec/*.tex)
BIBS=$(wildcard *.bib)
STYLES=$(wildcard *.sty)
CLASSES=$(wildcard *.cls)
BSTS=$(wildcard *.bst)

all: clean paper rebuttal

paper: $(MAIN_TEX).pdf

rebuttal: $(REBUTTAL_TEX).pdf

$(MAIN_TEX).pdf: $(MAIN_TEX).tex $(SECTIONS) $(BIBS) $(STYLES) $(CLASSES) $(BSTS)
	pdflatex $(MAIN_TEX).tex
	bibtex $(MAIN_TEX)
	pdflatex $(MAIN_TEX).tex
	pdflatex $(MAIN_TEX).tex

$(REBUTTAL_TEX).pdf: $(REBUTTAL_TEX).tex $(BIBS) $(STYLES) $(CLASSES) $(BSTS)
	@if [ -f "$(REBUTTAL_TEX).tex" ]; then \
		pdflatex $(REBUTTAL_TEX).tex; \
		bibtex $(REBUTTAL_TEX) || true; \
		pdflatex $(REBUTTAL_TEX).tex; \
		pdflatex $(REBUTTAL_TEX).tex; \
	fi

clean:
	find . -type f \( \
		-name "*.aux" -o -name "*.log" -o -name "*.bbl" -o -name "*.blg" \
		-o -name "*.out" -o -name "*.toc" -o -name "*.lof" -o -name "*.lot" \
		-o -name "*.gz" -o -name "*.nav" -o -name "*.snm" -o -name "*.fdb_latexmk" \
		-o -name "*.fls" -o -name "*.synctex.gz" \
	\) -delete

cleanall: clean
	find . -type f -name "*.pdf" -delete