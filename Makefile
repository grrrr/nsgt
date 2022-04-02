all: 
	python3 setup.py bdist_wheel
	python3 setup.py sdist
		
install:
	twine upload dist/*

test:
	python3 setup.py test
	twine check dist/*
	
