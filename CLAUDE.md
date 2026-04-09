# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Python library (PyPI: `tsoc-data-analysis`) for analyzing Cyprus (TSOC) power system operational data from Excel files. Provides a CLI (`tsoc-analyze`), a programmatic API (`execute()`, `extract_representative_ops()`), and modular utilities for load analysis, generator categorization, wind power analysis, data validation, and representative operating point extraction via K-means clustering.

## Build & Install

```bash
pip install -e .            # editable install
pip install -e ".[dev]"     # with test/lint deps
pip install -e ".[docs]"    # with Sphinx deps
```

## Testing

```bash
pytest                              # run all tests with coverage
pytest tests/test_system_configuration.py  # run one test file
pytest -k "test_files_structure"    # run a single test by name
pytest -m "not slow"                # skip slow tests
pytest -m "not integration"         # skip integration tests
```

Coverage is auto-enabled via pyproject.toml (`--cov=tsoc_data_analysis`). The end-to-end test (`test_end_to_end_representative_ops.py`) requires actual TSOC Excel data in `../raw_data`.

## Linting & Formatting

```bash
black src/ tests/           # format (line-length 88, target py38)
flake8 src/ tests/          # lint
mypy src/                   # type check (strict mode)
```

mypy is configured with strict settings (`disallow_untyped_defs`, `disallow_incomplete_defs`, etc.). External libs (matplotlib, seaborn, sklearn, scipy) have `ignore_missing_imports = true`.

## Architecture

Uses src-layout: all package code is under `src/tsoc_data_analysis/`.

**Module dependency flow:**

```
power_analysis_cli (orchestrator / CLI entry point)
  ├── excel_data_processor      → loads & cleans Excel files
  ├── power_data_validator      → validates & gap-fills time series
  ├── power_system_analytics    → calculates load, wind, reactive power metrics
  ├── power_system_visualizer   → generates matplotlib/seaborn plots
  └── system_configuration      → all constants, file mappings, validation params
operating_point_extractor       → K-means clustering for representative operating points
```

- **`system_configuration.py`** — Central config hub. All file mappings (`FILES`), column prefixes (`COLUMN_PREFIXES`), validation params (`DATA_VALIDATION`), clustering config (`REPRESENTATIVE_OPS`), and plot styling live here. Other modules import from this.
- **`power_analysis_cli.py`** — The `execute()` function is the primary API. The `PowerAnalysisCLI` class orchestrates the full pipeline: load Excel → validate → compute metrics → plot → export CSV. The `main()` function is the CLI entry point (`tsoc-analyze`).
- **`power_data_validator.py`** — `DataValidator` class with multiple validation strategies: type checks, limit checks, gap filling (linear/spline/KNN/polynomial), DST handling, anomaly detection (IQR, Z-score, Isolation Forest, LOF), rate-of-change validation, and power balance checks.
- **`operating_point_extractor.py`** — `extract_representative_ops()` uses K-means to select representative operating points. Handles MAPGL belt (critical low-load region), automatic cluster count selection, and quality metrics (silhouette, Calinski-Harabasz, Davies-Bouldin). `extract_representative_ops_enhanced()` adds preprocessing options.

## Key Patterns

- Configuration is centralized in `system_configuration.py` dictionaries, not scattered across modules. To change validation thresholds, file mappings, or clustering parameters, modify those dicts.
- The package expects TSOC Excel files with specific naming conventions and column structures defined in `FILES` and `COLUMN_PREFIXES`.
- Version is stored in `__init__.py` (`__version__`) and read dynamically by setuptools.
