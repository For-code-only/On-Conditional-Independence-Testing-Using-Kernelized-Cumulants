# CSIC

Code for **On Conditional Independence Testing Using Kernelized Cumulants**. The package implements CSIC and CSIC–CI, their HSIC and KCI-type comparators, and the three main experiments and the supplementary studies listed below.

## Installation

From the repository root, create a Python 3.12 environment and install the package.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Main experiments

```bash
# Small example
csic-run --quick --output results/quick

# Reproduce the unconditional, Gaussian conditional, and Seoul experiments
csic-run --output results/paper --workers 2
```

The full run writes both summary tables and all three figures to `results/paper/summary/`. To run a single study, add `--study unconditional`, `--study conditional`, or `--study seoul`. Rerunning the same command skips completed replications.

## Supplementary studies

Run `csic-study STUDY --output DIR` with the study name below. The default settings reproduce the paper's experiment design. Add `--quick` for a reduced example and use a separate output directory for each study and mode.

```bash
csic-study factorial --quick --output results/factorial-quick
```

| Study | Experiment | Paper reference |
| --- | --- | --- |
| `factorial` | Four combinations of kernel order and bandwidth with independent null calibration | Tables 5–7 and 10 |
| `ridge` | Common ridge selection and sensitivity to nuisance strength | Appendix E.3 |
| `oracle` | Fitted and exact conditional feature means at a shared reference center | Table 11 |
| `split-development` | Selection of test fractions on independent development data | Table 12 |
| `split` | Test-fraction selection followed by evaluation on fresh data | Tables 12–15 |
| `binary` | Binary responses at prevalences 0.25 and 0.49, cutoff sensitivity, and the unit-weight mixed kernel | Tables 16–18 |
| `balanced` | Independent binary-response experiment at prevalence 0.50 | Appendix E.6 |
| `weight` | Null pilot for selecting the continuous-response kernel weight | Appendix E.6 |
| `mixed` | Continuous-response mixed kernels on Gaussian and Seoul designs, with paired ROC comparisons | Table 19 and Appendix E.6 |

The split procedure adapts [Pogodin et al.](https://arxiv.org/abs/2402.13196).

## Testing your own data

```python
from csic import test

result = test(x, y, z, method="CSIC_CI", seed=123)
print(result["calibration"]["pvalue"])
```

Use `method="KCI"` for the conditional comparator. For unconditional tests, omit `z` and use `method="CSIC"` or `method="HSIC"`.

## Seoul data

`data/seoul_weather.npy` contains temperature, humidity, wind speed, solar radiation, and rainfall for all 8,760 rows of the [Seoul Bike Sharing Demand dataset](https://doi.org/10.24432/C5F62R). Each variable is standardized using its full-dataset mean and population standard deviation. The data are distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Tests

```bash
python -m unittest discover -s tests
```
