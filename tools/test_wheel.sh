#!/usr/bin/env bash
# Runs the Python tests against the wheel cibuildwheel has just installed,
# from a temporary directory so that the installed module is the one imported.
# On a free-threaded build it also checks, outside pytest, that importing the
# module leaves the GIL off.
#
# Usage: test_wheel.sh PROJECT

set -euo pipefail

PROJECT=$1
cd "$(mktemp -d)"

python -c "import meshsearch; print(meshsearch.__file__)"
pytest "$PROJECT/tests/test_meshgrid.py" "$PROJECT/tests/test_readme.py" \
  "$PROJECT/tests/test_determinism.py" -p no:cacheprovider -rs

if python -c "import sys; sys.exit(0 if hasattr(sys, '_is_gil_enabled') and not sys._is_gil_enabled() else 1)"; then
  echo "free-threaded build: GIL stays off"
  python -c "import sys, meshsearch; sys.exit(1 if sys._is_gil_enabled() else 0)"
fi
