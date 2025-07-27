# Makefile for CVPR LaTeX Submission

MAIN_TEX=cvpr
REBUTTAL_TEX=CvprRebuttal

all: paper

paper:
	pdflatex $(MAIN_TEX).tex
	bibtex $(MAIN_TEX)
	pdflatex $(MAIN_TEX).tex
	pdflatex $(MAIN_TEX).tex

rebuttal:
	pdflatex $(REBUTTAL_TEX).tex

clean:
	rm -f *.aux *.log *.bbl *.blg *.out *.toc *.lof *.lot *.gz *.nav *.snm *.fdb_latexmk *.fls

cleanall: clean
	rm -f $(MAIN_TEX).pdf $(REBUTTAL_TEX).pdf
