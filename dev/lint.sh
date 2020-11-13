autoflake -i -r --ignore-init-module-imports --remove-all-unused-imports .

# isort version check & run
ISORT_VERSION=$(isort --version-number)
if [[ "$ISORT_VERSION" != 5* ]]; then
    echo "error: linter requires isort=5"
    exit 1
fi
isort --sp ./setup.cfg .

black -l 88 -t py38 .
mypy --config-file ./setup.cfg . \
    | grep -v 'found module but no type hints or library stubs' \
    | grep -v 'note: See https://mypy.readthedocs.io/en/latest/running_mypy.html#missing-imports'
