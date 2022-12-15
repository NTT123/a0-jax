set -xeuo pipefail
readonly VENV_DIR=/tmp/a0-jax
rm -rf "${VENV_DIR}"
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python --version
pip install --upgrade pip
pip install black==22.3.0 pylint==2.15.8 mypy==0.991 pytest==7.2.0
pip install -r requirements.txt
black --diff --check $(git ls-files '*.py')
pylint --disable=all --enable=unused-import $(git ls-files '*.py')
mypy --ignore-missing-imports $(git ls-files '*.py')
pytest
set +u
deactivate
echo "All tests passed!"
