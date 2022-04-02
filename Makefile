.PHONY: all test upload

all:
	python3 setup.py bdist_wheel
	python3 setup.py sdist
	
test:
	python3 setup.py test
	twine check dist/*
	
upload:
	twine upload --skip-existing dist/*
