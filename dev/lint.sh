autoflake -i -r --ignore-init-module-imports --remove-all-unused-imports .
isort -sp ./setup.cfg -rc -y .
black -l 88 -t py38 .
mypy --config-file ./setup.cfg .
