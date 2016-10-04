# Make python file out of the ui files.

DESIGNERDIR = designer
BUILDDIR    = ui

# Commands

all: clean gui

gui:
	pyuic4 $(DESIGNERDIR)/gui.ui > $(BUILDDIR)/gui.py

clean:
	rm -rf $(BUILDDIR)/*
