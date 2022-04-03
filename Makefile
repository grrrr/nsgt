.PHONY: all test upload github

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

github:
	-git remote add github https://$(GITHUB_ACCESS_TOKEN)@github.com/$(GITHUB_USERNAME)/nsgt.git
	git push github
