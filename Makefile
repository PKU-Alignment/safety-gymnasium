print-%  : ; @echo $* = $($*)
PROJECT_NAME   = safety-gymnasium
COPYRIGHT      = "OmniSafe Team. All Rights Reserved."
PROJECT_PATH   = safety_gymnasium
SHELL          = /bin/bash
SOURCE_FOLDERS = $(PROJECT_PATH) examples tests docs
PYTHON_FILES   = $(shell find $(SOURCE_FOLDERS) -type f -name "*.py" -o -name "*.pyi")
COMMIT_HASH    = $(shell git log -1 --format=%h)
PATH           := $(HOME)/go/bin:$(PATH)
PYTHON         ?= $(shell command -v python3 || command -v python)
PYTESTOPTS     ?=

.PHONY: default
default: install

install:
	$(PYTHON) -m pip install -vvv .

install-editable:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install --upgrade setuptools
	$(PYTHON) -m pip install -vvv --editable .

install-e: install-editable  # alias

uninstall:
	$(PYTHON) -m pip uninstall -y $(PROJECT_NAME)

build:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install --upgrade setuptools wheel build
	$(PYTHON) -m build

# Tools Installation

check_pip_install = $(PYTHON) -m pip show $(1) &>/dev/null || (cd && $(PYTHON) -m pip install $(1) --upgrade)
check_pip_install_extra = $(PYTHON) -m pip show $(1) &>/dev/null || (cd && $(PYTHON) -m pip install $(2) --upgrade)

pylint-install:
	$(call check_pip_install_extra,pylint,pylint[spelling])
	$(call check_pip_install,pyenchant)

flake8-install:
	$(call check_pip_install,flake8)
	$(call check_pip_install,flake8-bugbear)
	$(call check_pip_install,flake8-comprehensions)
	$(call check_pip_install,flake8-docstrings)
	$(call check_pip_install,flake8-pyi)
	$(call check_pip_install,flake8-simplify)

py-format-install:
	$(call check_pip_install,isort)
	$(call check_pip_install_extra,black,black[jupyter])

ruff-install:
	$(call check_pip_install,ruff)

mypy-install:
	$(call check_pip_install,mypy)

pre-commit-install:
	$(call check_pip_install,pre-commit)
	$(PYTHON) -m pre_commit install --install-hooks

docs-install:
	$(call check_pip_install_extra,pydocstyle,pydocstyle[toml])
	$(call check_pip_install,doc8)
	$(call check_pip_install,sphinx)
	$(call check_pip_install,sphinx-autoapi)
	$(call check_pip_install,sphinx-autobuild)
	$(call check_pip_install,sphinx-copybutton)
	$(call check_pip_install,sphinx-autodoc-typehints)
	$(call check_pip_install,sphinx-design)
	$(call check_pip_install_extra,sphinxcontrib-spelling,sphinxcontrib-spelling pyenchant)
	$(PYTHON) -m pip install --requirement docs/requirements.txt

pytest-install:
	$(call check_pip_install,pytest)
	$(call check_pip_install,pytest-cov)
	$(call check_pip_install,pytest-xdist)

test-install: pytest-install
	$(PYTHON) -m pip install --requirement tests/requirements.txt

go-install:
	# requires go >= 1.16
	command -v go || (sudo apt-get install -y golang && sudo ln -sf /usr/lib/go/bin/go /usr/bin/go)

addlicense-install: go-install
	command -v addlicense || go install github.com/google/addlicense@latest

# Tests

pytest: test-install
	cd tests && $(PYTHON) -c 'import $(PROJECT_PATH)' && \
	$(PYTHON) -m pytest --verbose --color=yes --durations=0 \
		--cov="$(PROJECT_PATH)" --cov-config=.coveragerc --cov-report=xml --cov-report=term-missing \
		$(PYTESTOPTS) .

test: pytest

# Python linters

pylint: pylint-install
	$(PYTHON) -m pylint $(PROJECT_PATH)

flake8: flake8-install
	$(PYTHON) -m flake8 --count --show-source --statistics --exclude=safety_gymnasium/tasks/safe_isaac_gym

py-format: py-format-install
	$(PYTHON) -m isort --project $(PROJECT_PATH) --check $(PYTHON_FILES) && \
	$(PYTHON) -m black --check $(PYTHON_FILES)

ruff: ruff-install
	$(PYTHON) -m ruff check --exclude=safety_gymnasium/tasks/safe_isaac_gym/ .

ruff-fix: ruff-install
	$(PYTHON) -m ruff check . --fix --exit-non-zero-on-fix

mypy: mypy-install
	$(PYTHON) -m mypy $(PROJECT_PATH) --install-types --non-interactive

pre-commit: pre-commit-install
	$(PYTHON) -m pre_commit run --all-files

# Documentation

addlicense: addlicense-install
	addlicense -c $(COPYRIGHT) \
	-ignore tests/coverage.xml \
	-ignore safety_gymnasium/tasks/safe_isaac_gym/envs/assets/dataset/one_drawer_cabinet/40147_link_1/door.yaml \
	-ignore safety_gymnasium/tasks/safe_isaac_gym/envs/assets/dataset/one_drawer_cabinet/40147_link_1/handle.yaml \
	-ignore safety_gymnasium/tasks/safe_isaac_gym/envs/assets/dataset/one_drawer_cabinet/40147_link_1/tree_hier_after_merging.html \
	-ignore safety_gymnasium/tasks/safe_isaac_gym/envs/assets/dataset/one_drawer_cabinet/40147_link_1/tree_hier.html \
	-ignore safety_gymnasium/tasks/safe_isaac_gym/envs/assets/urdf/shadow_hand_description/meshes/convert_dae2obj.py \
	-ignore safety_gymnasium/tasks/safe_isaac_gym/utils/util.py \
	-ignore safety_gymnasium/tasks/safe_isaac_gym/utils/Logger.py \
	-ignore safety_gymnasium/tasks/safe_isaac_gym/utils/process_sarl.py \
	-ignore docs/_build/spelling/_sphinx_design_static/design-tabs.js \
	-ignore safety_gymnasium/tasks/safe_isaac_gym/envs/tasks/freight_franka_close_drawer.py \
	-ignore safety_gymnasium/tasks/safe_isaac_gym/envs/tasks/freight_franka_pick_and_place.py \
	-ignore docs/_build/spelling/_sphinx_design_static/design-style.1e8bd061cd6da7fc9cf755528e8ffc24.min.css \
	-l apache -y 2022-$(shell date +"%Y") $(SOURCE_FOLDERS)

docstyle: docs-install
	make -C docs clean
	$(PYTHON) -m pydocstyle $(PROJECT_PATH) && doc8 docs && make -C docs html SPHINXOPTS="-W"

docs: docs-install
	$(PYTHON) -m sphinx_autobuild --watch $(PROJECT_PATH) --open-browser docs docs/build

spelling: docs-install
	make -C docs clean
	make -C docs spelling SPHINXOPTS="-W"

clean-docs:
	make -C docs clean

# Utility functions

# TODO: add mypy when ready
# TODO: add docstyle when ready
lint: ruff flake8 py-format pylint addlicense spelling

format: py-format-install ruff-install addlicense-install
	$(PYTHON) -m isort --project $(PROJECT_PATH) $(PYTHON_FILES)
	$(PYTHON) -m black $(PYTHON_FILES)
	$(PYTHON) -m ruff check . --fix --exit-zero
	addlicense -c $(COPYRIGHT) \
	-ignore tests/coverage.xml \
	-ignore safety_gymnasium/tasks/safe_isaac_gym/envs/assets/dataset/one_drawer_cabinet/40147_link_1/door.yaml \
	-ignore safety_gymnasium/tasks/safe_isaac_gym/envs/assets/dataset/one_drawer_cabinet/40147_link_1/handle.yaml \
	-ignore safety_gymnasium/tasks/safe_isaac_gym/envs/assets/dataset/one_drawer_cabinet/40147_link_1/tree_hier_after_merging.html \
	-ignore safety_gymnasium/tasks/safe_isaac_gym/envs/assets/dataset/one_drawer_cabinet/40147_link_1/tree_hier.html \
	-ignore safety_gymnasium/tasks/safe_isaac_gym/envs/assets/urdf/shadow_hand_description/meshes/convert_dae2obj.py \
	-ignore safety_gymnasium/tasks/safe_isaac_gym/utils/util.py \
	-ignore safety_gymnasium/tasks/safe_isaac_gym/utils/Logger.py \
	-ignore safety_gymnasium/tasks/safe_isaac_gym/utils/process_sarl.py \
	-ignore docs/_build/spelling/_sphinx_design_static/design-tabs.js \
	-ignore safety_gymnasium/tasks/safe_isaac_gym/envs/tasks/freight_franka_close_drawer.py \
	-ignore safety_gymnasium/tasks/safe_isaac_gym/envs/tasks/freight_franka_pick_and_place.py \
	-ignore docs/_build/spelling/_sphinx_design_static/design-style.1e8bd061cd6da7fc9cf755528e8ffc24.min.css \
	-l apache -y 2022-$(shell date +"%Y") $(SOURCE_FOLDERS)

clean-py:
	find . -type f -name  '*.py[co]' -delete
	find . -depth -type d -name "__pycache__" -exec rm -r "{}" +
	find . -depth -type d -name ".ruff_cache" -exec rm -r "{}" +
	find . -depth -type d -name ".mypy_cache" -exec rm -r "{}" +
	find . -depth -type d -name ".pytest_cache" -exec rm -r "{}" +
	rm tests/.coverage
	rm tests/coverage.xml

clean-build:
	rm -rf build/ dist/
	rm -rf *.egg-info .eggs

clean: clean-py clean-build clean-docs
