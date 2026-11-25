"""Tests of the Python code shown in README.md.

The README's Python block is meant to run as copied, in a fresh interpreter
state with nothing defined beforehand, so it is executed here in a namespace of
its own. The C++ block is compiled and run by ctest (test
readme_cpp), since that needs the library and a compiler.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def python_blocks():
    text = (ROOT / "README.md").read_text()
    return re.findall(r"```python\n(.*?)```", text, re.S)


def test_the_readme_has_the_python_block():
    assert len(python_blocks()) == 1


def test_the_python_block_runs_as_copied():
    block = python_blocks()[0]
    exec(compile(block, "README.md, python block", "exec"), {"__name__": "readme"})


# The block's comments promise what each line returns; these check the
# promises that a reader would rely on when copying it.
def test_the_python_block_shows_what_it_says():
    namespace = {}
    exec(compile(python_blocks()[0], "README.md", "exec"), namespace)
    grid, nearest = namespace["grid"], namespace["nearest"]
    assert len(grid) == 1000
    assert len(grid.nearest_objects(10, 0.5, 0.5, 0.5)) == 10
    assert grid.count_objects(0.5, 0.5, 0.5, 0.15) == grid.close_objects(0.5, 0.5, 0.5, 0.15).size
    assert nearest.shape == (200,) and nearest.dtype.name == "uint32"
