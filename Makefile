.PHONY: all test build_bdist build_sdist test_nsgt test_dist upload_pypi push_github

# all builds
all: build_bdist build_sdist

# all tests
test: test_nsgt test_dist
	
# build binary dist
# resultant *.whl file will be in subfolder dist
build_bdist:
	python3 setup.py bdist_wheel
	# for linux use auditwheel to convert to manylinux format
	if auditwheel repair dist/nsgt*.whl; then \
		rm dist/nsgt*.whl; \
		mv wheelhouse/nsgt*.whl dist/; \
	fi
	
# build source dist
# resultant file will be in subfolder dist
build_sdist:
	python3 setup.py sdist

# test python module
test_nsgt:
	python3 setup.py test

# test packages
test_dist:
	twine check dist/*

# upload to pypi
upload_pypi:
	twine upload --skip-existing dist/*

# push to github
push_github:
	-git remote add github https://$(GITHUB_ACCESS_TOKEN)@github.com/$(GITHUB_USERNAME)/nsgt.git
	# we need some extra treatment because the gitlab-runner doesn't check out the full history
	git push github HEAD:master --tags
