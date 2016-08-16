# Make python file out of the ui files.

all: clean gui

gui:
	pyuic4 designer/gui.ui > gui.py

game:
	pyuic4 designer/game.ui > game.py

clean:
	rm -r __pycache__
	rm gui.py
