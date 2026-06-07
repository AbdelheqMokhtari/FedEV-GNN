.PHONY: format lint test check release

format:
	black src tests
	isort src tests

lint:
	flake8 src tests

test:
	pytest

check:
	pre-commit run --all-files

release:
	pre-commit run --all-files
	cz bump
	git checkout main
	git merge develop
	git push
	git push --tags
	git checkout develop
