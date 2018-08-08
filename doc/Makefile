# Makefile for Sphinx documentation
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
PAPER         =
BUILDDIR      = _build
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .

.PHONY: clean html

html: clean
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	touch $(BUILDDIR)/html/.nojekyll
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

clean:
	-rm -rf $(BUILDDIR)
	-rm -rf auto_examples
	-rm -rf generated
	-rm -rf modules
