"""Sphinx configuration for the meshsearch documentation.

The C++ reference comes from Doxygen XML through Breathe, the Python reference
from the docstrings of the installed module through autodoc. Both are generated
from the same source the library is built from, so neither can drift from it
without the build saying so.

    pip install -r docs/requirements.txt
    pip install .
    sphinx-build -W --keep-going docs docs/_build/html
"""

import pathlib
import subprocess
import tomllib

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent

project = "meshsearch"
author = "Simone Sartori"
copyright = "2026, Simone Sartori"
release = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
version = release

# Doxygen runs as part of the Sphinx build, so there is one command to remember
# and no way to publish a C++ reference built from a stale header.
subprocess.run(["doxygen", "Doxyfile"], cwd=HERE, check=True)

extensions = [
    "breathe",
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
]

breathe_projects = {"meshsearch": str(HERE / "doxygen" / "xml")}
breathe_default_project = "meshsearch"
breathe_domain_by_extension = {"h": "cpp"}

autodoc_member_order = "groupwise"
autodoc_default_options = {"members": True}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

myst_enable_extensions = ["colon_fence", "attrs_block"]
myst_heading_anchors = 3

templates_path = ["_templates"]
exclude_patterns = ["_build", "doxygen", "requirements.txt", "README.md"]

html_theme = "furo"
html_title = f"meshsearch {release}"
html_static_path = []
