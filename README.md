# CSIC

Code for **On Conditional Independence Testing Using Kernelized Cumulants**  

Includes CSIC / CSIC–CI, HSIC / KCI-type comparators, and the unconditional,
Gaussian conditional and Seoul main experiments.

## Installation

Python 3.12. Run from the repository directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Experiments

```bash
# Quick example
csic-run --quick --output results/quick

# Main experiments
csic-run --output results/paper --workers 2
```

For a single study, add `--study unconditional`, `--study conditional` or
`--study seoul`. Results and figures are saved in `OUTPUT/summary/`.
Running the same command resumes unfinished work. See `csic-run --help` for options.

Plot the supplied main results without rerunning the experiments:

```bash
csic-plot --rates results/reference/rates.csv --output results/figures
```

The defaults match the paper: `n = 200, 400, 800`; Gaussian `D = 2, 5`;
1,000 null and 500 repetitions per alternative; `B = 4999`, `alpha = 0.05`.
The Y bandwidth is the median distance for HSIC / KCI-type and twice the median
for CSIC / CSIC–CI. Both conditional methods use
`rho_n = 0.01 * log(n) / log(400)`, with `lambda_n = rho_n / n`.

## Testing your data

```python
from csic import test

result = test(x, y, z, method="CSIC_CI", seed=123)
print(result["calibration"]["pvalue"])
```

Use `method="KCI"` for the conditional comparator; omit `z` for `"CSIC"` or
`"HSIC"`. KCI-type is the matched implementation used in the paper.

## Files and data

`src/csic/` contains the algorithms, generators, runner and plotting code.
`results/reference/` contains the main result tables; `tests/` contains checks
runnable with `python -m unittest discover -s tests`.

The Seoul semi-synthetic experiment uses the bundled five weather variables from
[Seoul Bike Sharing Demand (UCI, 2020)](https://doi.org/10.24432/C5F62R), under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). All 8,760 rows are retained;
temperature, humidity, wind speed, solar radiation and rainfall are standardized
using the archive means and population standard deviations.
