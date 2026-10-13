# Standardized Real-World Change Detection Data (SRWCDD)

Code and results for my bachelor's thesis at the Institute for Program Structures and Data Organization (IPD), Department of Informatics, Karlsruhe Institute of Technology (KIT), written from May to September 2022. The thesis introduces RMMDCD (Recursive MMD Change Point Detection) and uses it to relabel the data sets of the Turing Change Point Detection Benchmark.

| | |
|---|---|
| Author | Moritz Teichner |
| Reviewer | Prof. Dr. Klemens Böhm |
| Advisor | M.Sc. Florian Kalinke |

## Abstract

The reliable detection of change points is a fundamental task across many fields, e.g., in finance, bioinformatics, and medicine. To define "change points", we assume that there is a distribution generating the data we observe. This distribution may change over time. Then, a change point is a change in this underlying distribution, i.e., the distribution coming before a change point is different from the distribution coming after. The principled way to detect such changes thus is to compare the distributions with a statistical test.

While change point detection is an unsupervised problem in practice, i.e., the data is unlabeled, developing and evaluating data analysis algorithms requires labeled data. However, only a few labeled real-world data sets are publicly available; many are relatively small or have ambiguous labels. Further, optimizing parameters on a few data sets may lead to overfitting [1]. To address these issues, [1] publishes 37 data sets annotated by data scientists and machine learning researchers and assesses fourteen change detection algorithms on these. Yet, concerns remain because these were labeled by hand: Can humans correctly identify changes according to the definition, and if so, can they do so consistently?

We tackle these questions by introducing a novel change point detection algorithm, dubbed RMMDCD, which we use to relabel existing data sets. To this end, we leverage a non-parametric hypothesis test that builds on Maximum Mean Discrepancy (MMD), i.e., our algorithm identifies changes in a principled manner. RMMDCD operates in an offline, multivariate and non-parametric setting and detects arbitrary changes. Under the assumption that the existing benchmark data sets have been labeled correctly, we evaluate our results w.r.t. the existing annotations on these data sets. On multivariate data, we outperform all applicable algorithms of the existing benchmark study [1]. On univariate data, we achieve competitive results as well. To measure the consistency of the human annotations, we compare them to the new labels with the F1 score: Here, we find that the human annotators are identifying changes consistent with each other. To assess the influence of the new and principled annotations, we use them to reevaluate the algorithms of [1] and compare the respective results. From our results, we conclude that the performance of these algorithms does not change by a significant margin. This conclusion is in line with the previous observation, as the human annotators identified the changes consistently, and the labels did not change significantly.

## Method

RMMDCD performs binary segmentation with a kernel two-sample test:

1. For every split point `i` that leaves at least `min_win_size` samples on each side, compute the quadratic-time MMD estimate between `X[:i]` and `X[i:]` with a Gaussian kernel.
2. Take the split with the largest MMD and test it for significance, using either
   - the analytic bound on the MMD under the null hypothesis (`alpha=True`), or
   - a permutation test with 400 permutations (`alpha=False`).
3. If the change is significant, record it and recurse on the left and right segments.

The kernel bandwidth `gamma` comes from the median heuristic, evaluated on 100 random sub-windows of the series and multiplied by a tunable factor. Keeping the smallest value favours large, "global" changes over minor local ones. Because the sub-windows are random, repeated runs can differ slightly; call `np.random.seed(...)` first for repeatable output.

## Repository layout

```
rmmdcm.py               RMMDCD, the F1 metric, and the grid-search driver
Results/new_labels.json Change points RMMDCD assigns to each TCPD data set (thesis Table A.1)
Results/new_scores.json F1 score of those labels against the human annotations (thesis Table A.1)
```

Index `0` appears in every label list because the TCPDBench F1 metric treats the start of the series as a change point.

## Installation

Requires Python 3.8 or newer.

```bash
git clone https://github.com/moritzteichner/SRWCDD.git
cd SRWCDD
pip install -r requirements.txt
```

## Data

The experiments use the Turing Change Point Dataset (TCPD) [1]:

1. Build the data sets with the instructions in [alan-turing-institute/TCPD](https://github.com/alan-turing-institute/TCPD). This produces one `<name>.json` file per series.
2. Download the human annotations, `analysis/annotations/annotations.json`, from [alan-turing-institute/TCPDBench](https://github.com/alan-turing-institute/TCPDBench).

## Usage

Run the full grid search over all data sets:

```bash
python rmmdcm.py --data-dir path/to/tcpd --annotations path/to/annotations.json --out results.csv
```

The grid follows Table 5.1 of the thesis:

| Parameter | Values |
|---|---|
| significance level α (%) | 10⁻⁶, 10⁻⁵, 10⁻⁴, 10⁻³, 10⁻², 0.1, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 55, 60, 70, 75 |
| minimum window size | 20, 30, 40, 50 |
| `gamma` factor | 0.001, 0.0018, 0.0055, 0.01, 0.018, 0.055, 0.1, 0.18, 0.55, 1, 1.8, 5.5, 10, 55, 100 |
| threshold | analytic bound, permutation test |

Each row of `results.csv` holds the parameter combination, the data set, its F1 score, and the detected change points.

To label a single data set with the parameters the thesis selected (Table 5.2):

```python
from rmmdcm import calculate_f1

# univariate data: alpha = 55 %, mws = 30, gamma factor = 0.018, analytic threshold
f1, change_points = calculate_f1("nile", gamma_factor=0.018, mws=30, los=55, alpha=True,
                                 data_dir="path/to/tcpd", annotations="path/to/annotations.json")

# multivariate data: alpha = 40 %, mws = 30, gamma factor = 1, analytic threshold
f1, change_points = calculate_f1("occupancy", gamma_factor=1, mws=30, los=40, alpha=True,
                                 data_dir="path/to/tcpd", annotations="path/to/annotations.json")
```

## Results

### Selected parameters

The grid search picked the parameters with the best mean F1 score against the human annotations (thesis Table 5.2):

| Data | α (%) | mws | `gamma` factor | Threshold |
|---|---|---|---|---|
| Univariate (32 data sets) | 55 | 30 | 0.018 | analytic |
| Multivariate (4 data sets) | 40 | 30 | 1 | analytic |

### Agreement with the human annotations

The labels in [`Results/`](Results/) were produced with these parameters. They cover 36 of the 37 TCPD data sets; `uk_coal_employ` is excluded because of missing values.

| Data | Mean F1 |
|---|---|
| Univariate | 0.691 |
| Multivariate | 0.660 |

### Key findings

- On multivariate data, RMMDCD ranks first among the applicable algorithms of the original benchmark study (default parameters). On univariate data it ranks fourth, and the gap to the top three is not statistically significant.
- Taking the RMMDCD labels as ground truth, the individual human annotators reach mean F1 scores between 0.49 and 0.74 with variances of at most 0.078. The thesis concludes that the annotators label consistently.
- Re-running the benchmark with the RMMDCD labels changes which algorithm ranks first, while the relative performance of the algorithms changes only by non-significant margins. Recommended algorithms under the new labels: AMOC (univariate, default), RFPOP (univariate, oracle), ECP (multivariate, default), BOCPD (multivariate, oracle).

<details>
<summary>Per-data-set F1 scores</summary>

| Data set | F1 | Data set | F1 |
|---|---|---|---|
| apple | 0.594 | jfk_passengers | 0.723 |
| bank | 1.000 | lga_passengers | 0.535 |
| bee_waggle_6 | 0.929 | measles | 0.947 |
| bitcoin | 0.372 | nile | 1.000 |
| brent_spot | 0.521 | occupancy | 0.595 |
| businv | 0.588 | ozone | 0.723 |
| centralia | 0.763 | rail_lines | 0.537 |
| children_per_woman | 0.405 | ratner_stock | 0.724 |
| co2_canada | 0.691 | robocalls | 0.636 |
| construction | 0.696 | run_log | 0.521 |
| debt_ireland | 0.469 | scanline_126007 | 0.925 |
| gdp_argentina | 0.824 | scanline_42049 | 0.668 |
| gdp_croatia | 0.824 | seatbelts | 0.474 |
| gdp_iran | 0.652 | shanghai_license | 0.774 |
| gdp_japan | 0.889 | unemployment_nl | 0.566 |
| global_co2 | 0.595 | us_population | 0.889 |
| homeruns | 0.675 | usd_isk | 0.489 |
| iceland_tourism | 0.947 | well_log | 0.602 |

</details>

## Acknowledgements

- The MMD estimator, Gaussian kernel, median heuristic, and permutation test are adapted from Heiko Strathmann's [kernel testing tutorial](https://github.com/karlnapf/ds3_kernel_testing/blob/master/solutions.ipynb).
- The F1 metric follows Florian Kalinke's implementation in [FlopsKa/StreamDatasets](https://github.com/FlopsKa/StreamDatasets/blob/main/changeds/metrics.py), which in turn follows TCPDBench.

## License

No license is granted. All rights are reserved by the author. If you want to use, modify, or redistribute this code or the results, please open an issue or contact me through GitHub to ask for permission.

## References

[1] G. J. J. van den Burg and C. K. I. Williams. *An Evaluation of Change Point Detection Algorithms.* arXiv:2003.06222, 2020.
