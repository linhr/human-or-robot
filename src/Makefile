DATA_PATH = ../data
WORKSPACE_PATH = ../workspace

.PHONY: all clean distclean

all:
	python features.py
	python prediction.py

clean:
	rm -f *.pyc

distclean: clean
	rm -rf $(WORKSPACE_PATH)
