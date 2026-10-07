# Benchmarks

These scripts reproduce the numbers on the documentation's performance page,
`docs/benchmark.md`. They are not part of the package: they are not installed,
and nothing in the library or its tests uses them.

- `benchmark.py` compares meshsearch with scipy's `cKDTree` and sklearn's
  `BallTree`.
- `benchmark_cellsize.py` measures what the cell side costs. Run it on your
  own data to choose one.

The docstring at the top of each script says how to run it.

## Dependencies

Besides an installed `meshsearch`, the scripts need:

- `numpy` and `psutil`, for both scripts;
- `scipy` and `scikit-learn`, for `benchmark.py` only.

The package's `benchmark` extra installs all of these:

```
pip install ".[benchmark]"
```
