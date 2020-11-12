autoflake -i -r --ignore-init-module-imports --remove-all-unused-imports .
# for isort<5.0.0: isort -sp ./setup.cfg -rc -y .
isort --sp ./setup.cfg .
black -l 88 -t py38 .
mypy --config-file ./setup.cfg . | grep -v 'found module but no type hints or library stubs'
