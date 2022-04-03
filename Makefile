.PHONY: all test build_bdist build_sdist test_nsgt test_dist upload_pypi push_github

all: build_bdist build_sdist

test: test_nsgt test_dist
	
build_bdist:
	python3 setup.py bdist_wheel
	auditwheel repair dist/nsgt*.whl
	rm dist/nsgt*.whl
	mv wheelhouse/nsgt*.whl dist/
	
build_sdist:
	python3 setup.py sdist

test_nsgt:
	python3 setup.py test
	
test_dist:
	twine check dist/*

upload_pypi:
	twine upload --skip-existing dist/*

push_github:
	-git remote add github https://$(GITHUB_ACCESS_TOKEN)@github.com/$(GITHUB_USERNAME)/nsgt.git
	git push github
