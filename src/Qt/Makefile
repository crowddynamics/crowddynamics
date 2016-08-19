# Make python file out of the ui files.

DESIGNERDIR = designer
BUILDDIR    = ui

# Commands

all: clean gui game

gui:
	pyuic4 $(DESIGNERDIR)/gui.ui > $(BUILDDIR)/gui.py

game:
	pyuic4 $(DESIGNERDIR)/game.ui > $(BUILDDIR)/game.py

clean:
	rm -rf $(BUILDDIR)/*
