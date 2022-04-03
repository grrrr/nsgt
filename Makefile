.PHONY: all test upload

all:
	python3 setup.py bdist_wheel
	auditwheel repair dist/nsgt*.whl
	rm dist/nsgt*.whl
	mv wheelhouse/nsgt*.whl dist/
	python3 setup.py sdist
	
test:
	python3 setup.py test
	twine check dist/*
	
upload:
	twine upload --skip-existing dist/*
