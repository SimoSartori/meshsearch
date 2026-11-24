"""Tests of the Python code shown in README.md.

Each Python block of the README is meant to run as copied, in a fresh
interpreter state with nothing defined beforehand, so each is executed here in
a namespace of its own. The C++ block is compiled and run by ctest (test
readme_cpp), since that needs the library and a compiler.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def python_blocks():
    text = (ROOT / "README.md").read_text()
    return re.findall(r"```python\n(.*?)```", text, re.S)


def test_the_readme_has_the_python_blocks():
    assert len(python_blocks()) == 2


@pytest.mark.parametrize("number", range(2))
def test_a_python_block_runs_as_copied(number):
    block = python_blocks()[number]
    exec(compile(block, f"README.md, python block {number + 1}", "exec"), {"__name__": "readme"})


def test_the_first_block_shows_what_it_says():
    namespace = {}
    exec(compile(python_blocks()[0], "README.md", "exec"), namespace)
    grid, box = namespace["grid"], namespace["box"]
    assert len(grid) == 1000
    assert grid.periodic == (False, False, False)
    assert box.periodic == (True, True, True)
    assert len(namespace["grid"].nearest_objects(10, 0.5, 0.5, 0.5)) == 10


def test_the_second_block_shows_what_it_says():
    namespace = {}
    exec(compile(python_blocks()[1], "README.md", "exec"), namespace)
    assert namespace["nearest"].shape == (200,)
    assert namespace["five"].shape == (200, 5)
    assert len(namespace["off"]) == 201
    assert namespace["counts"].dtype.name == "int64"
    assert namespace["alive"].all()
