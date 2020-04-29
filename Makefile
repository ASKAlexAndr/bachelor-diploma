all: build run

build:
	git submodule init
	git submodule update --remote
	latexmk -xelatex -synctex=1 -jobname=thesis main.tex

run:
	chrome thesis.pdf

clean:
	rm *.aux \
	*.fdb_latexmk \
	*.fls \
	*.lof \
	*.lot \
	*.log \
	*.out \
	*.synctex.gz \
	*.toc