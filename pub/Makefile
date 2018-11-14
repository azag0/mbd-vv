BLDDIR = _build
export max_print_line = 1000000

.PRECIOUS: %.tex

default: preprint

preprint: $(BLDDIR)/preprint.pdf

tex: preprint.tex preamble.tex si.tex

si: $(BLDDIR)/si.pdf

diff: $(BLDDIR)/diff.pdf

all: preprint si figs

figs:
	@mkdir -p $(BLDDIR)
	pdflatex -output-directory=$(BLDDIR) -shell-escape export-figures.tex

$(BLDDIR)/%.pdf: %.tex refs-zotero.bib FORCE
	latexmk -pdf -outdir=$(BLDDIR) -interaction=nonstopmode $<

$(BLDDIR)/preprint.pdf: preamble.tex
$(BLDDIR)/si.pdf: preamble.tex

%.tex: metadata.yml %.tex.in
	./texgen.py $< $@ <$(word 2,$^)

bib: refs-zotero.bib

refs-zotero.bib: FORCE
	-curl -sfo $@ "http://localhost:23119/better-bibtex/collection?/0/References/mbd-vv.bibtex&exportNotes=false&useJournalAbbreviation=true" && gsed -i '/./,$$!d' $@

clean:
	rm -rf $(BLDDIR)

distclean: clean
	rm -f $(patsubst %.tex.in,%.tex,$(wildcard *.tex.in))

FORCE:
