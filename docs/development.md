# Development

## Tests

The tests are in the `tests` folder and run with [pytest](https://docs.pytest.org/). With the environment active,
from the repository folder:

```bash
pytest
```

They test Curve2Flood's functions on small made-up grids and tables, and the FLDPLN library solvers on a few rasters
of a real river (`tests/data/scottsbluff`). There the iterative solver must match `tests/fldpln_model_v5ram.py`, a
Python port of the University of Kansas's MATLAB FLDPLN model, bit for bit, one segment at a time and through a whole
library build (to the centimetre the library keeps), spilling to a steady state or once a step. Where MATLAB is on
the `PATH`, the port must in turn match the MATLAB model itself (`tests/fldpln_matlab`) bit for bit, on the same
segments but one whose spill reaches the raster's edge, where the MATLAB model stops with an error; elsewhere, as on
GitHub, that test is skipped. They take under a minute, MATLAB included.

GitHub runs the tests on every pull request (`.github/workflows/tests.yml`), on Linux and Windows, in a new conda
environment made from `environment.yml`. It also builds the documentation with `mkdocs build --strict`, which fails
on any warning, such as a broken link.

## Adding a setting

1. Add it to `Config` (`curve2flood/config.py`): a field named for its input-file key in lower case, with its type
   and default. The type converts the text of a text input file.
2. If a run can't do without it, check for it in `Config.validate_options`, so a missing value stops the run with a
   clear error.
3. Document it in `docs/input_file.md`.
4. Add a test (`tests/test_config.py`).

ARC reads the same input files and rejects keys it doesn't know, except those it lists as Curve2Flood's
(`_CURVE2FLOOD_KEYS` in ARC's `arc/config.py`), so add a new key there too. Likewise, a new ARC key goes in
`_ARC_KEYS` here; `tests/test_config.py` checks this when ARC is installed.

## Documentation

The documentation is built with [MkDocs](https://www.mkdocs.org/) from `docs/`. To see it as you edit:

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

Read the Docs builds it from `docs/` (see `.readthedocs.yaml`).

The diagrams are [Mermaid](https://mermaid.js.org/) flowcharts, written in the pages as `mermaid` code blocks and
drawn by the browser. The figures in `docs/figures` are small made-up cross sections drawn by
`docs/figures/make_figures.py`; run it again after changing it:

```bash
python docs/figures/make_figures.py
```

## Benchmarks

`benchmarks/fldpln_matlab/run_benchmark.py` times the FLDPLN library build, and measures its memory, against the
University of Kansas's MATLAB model, and writes the table on the [FLDPLN page](fldpln.md#speed-and-memory). The
script's docstring says how it measures.
