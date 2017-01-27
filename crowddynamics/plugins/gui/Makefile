# Make python file out of the ui files.

SOURCEDIR = designer
BUILDDIR  = ui


.PHONY: all
all:
	clean gui

.PHONY: clean
clean:
	rm -rf $(BUILDDIR)/*

.PHONY: gui
gui:
	pyuic4 $(SOURCEDIR)/gui.ui > $(BUILDDIR)/gui.py

