all: build

build:
	latexmk -xelatex -synctex=1 -jobname=presentation presentation.tex

run: build clean
	chrome presentation.pdf

clean:
	rm *.aux \
	*.fdb_latexmk \
	*.fls \
	*.log \
	*.out \
	*.synctex.gz \
	*.toc \
	*.snm \
	*.nav