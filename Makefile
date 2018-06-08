SHELL=/bin/bash

install:
	pip install -U pip
	pip install -r requirements.txt
	pip install .

clean:
	find . -name "*.py[co]" -delete

tests:
	python -m unittest discover -v

test: tests