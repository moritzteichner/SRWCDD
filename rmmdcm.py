"""RMMDCD: Recursive Maximum Mean Discrepancy Change Point Detection.

Code for the bachelor's thesis "Standardized Real-World Change Detection Data"
(Moritz Teichner, KIT, 2022).

The module contains three parts:

1. Kernel two-sample testing utilities (squared distances, Gaussian kernel,
   MMD estimator, median heuristic, permutation test).
2. The RMMDCD algorithm (thesis Algorithm 2): find the split with the largest
   MMD, test it for significance, and recurse on both halves.
3. Evaluation: the multi-annotator F1 score of TCPDBench and a grid search over
   the RMMDCD parameters on the TCPD data sets.
"""

# Imports

import argparse
import csv
import json
import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist, cdist


# --------- Underlying functionality ---------

# ------ Copied from https://github.com/karlnapf/ds3_kernel_testing/blob/master/solutions.ipynb ------

def sq_distances(X, Y=None):
    """Pairwise squared Euclidean distances.

    X has shape (n, d) and Y has shape (m, d). Returns an (n, n) matrix if Y is
    None, otherwise an (n, m) matrix.
    """
    assert (X.ndim == 2)

    if Y is None:
        sq_dists = squareform(pdist(X, 'sqeuclidean'))
    else:
        assert (Y.ndim == 2)
        assert (X.shape[1] == Y.shape[1])
        sq_dists = cdist(X, Y, 'sqeuclidean')

    return sq_dists


def gauss_kernel(X, Y=None, gamma=1.0):
    """Gaussian kernel matrix k(x, y) = exp(-gamma * ||x - y||^2) (thesis Eq. 4.11)."""
    sq_dists = sq_distances(X, Y)
    K = np.exp(-gamma * sq_dists)
    return K


def quadratic_time_mmd(X, Y, kernel):
    """Estimate of the squared MMD between the samples X (n, d) and Y (m, d).

    The within-sample terms leave out the diagonal (k(x_i, x_i)), which makes
    them unbiased averages over distinct pairs. Runs in O((n + m)^2).
    """
    assert X.ndim == Y.ndim == 2
    K_XX = kernel(X, X)
    K_XY = kernel(X, Y)
    K_YY = kernel(Y, Y)

    n = len(K_XX)
    m = len(K_YY)

    # Drop the k(x_i, x_i) terms so each within-sample mean runs over i != j
    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)
    mmd = np.sum(K_XX) / (n * (n - 1)) + np.sum(K_YY) / (m * (m - 1)) - 2 * np.sum(K_XY) / (n * m)
    return mmd


def gaussian_kernel_median_heuristic(Z):
    """Median heuristic for the Gaussian kernel: gamma = 1 / median pairwise distance in Z."""
    sq_dists = sq_distances(Z)
    # Ignore the zero distance of every point to itself
    np.fill_diagonal(sq_dists, np.nan)
    sq_dists = np.ravel(sq_dists)
    sq_dists = sq_dists[~np.isnan(sq_dists)]
    median_dist = np.median(np.sqrt(sq_dists))
    return (1 / median_dist)


def two_sample_permutation_test(test_statistic, X, Y, num_permutations, prog_bar=True):
    """Sample the null distribution of a two-sample statistic by permutation.

    The samples X and Y are pooled, shuffled and split again into parts of the
    original sizes. Under the null hypothesis (same distribution) every such
    split is equally likely, so the returned statistics approximate the null
    distribution (thesis Section 3.4.2).
    """
    statistics = np.zeros(num_permutations)

    range_ = range(num_permutations)
    # if prog_bar:
    # range_ = tqdm(range_)
    for i in range_:
        # concatenate samples
        if X.ndim == 1:
            Z = np.hstack((X, Y))
        elif X.ndim == 2:
            Z = np.vstack((X, Y))

        # Shuffle the pooled sample and split it at the original boundary
        perm_inds = np.random.permutation(len(Z))
        Z = Z[perm_inds]
        X_ = Z[:len(X)]
        Y_ = Z[len(X):]
        my_test_statistic = test_statistic(X_, Y_)
        statistics[i] = my_test_statistic
    return statistics


# ------ Copied End------

def find_max_mmd(X, min_win_size, kernel):
    """Find the split index i that maximises MMD(X[:i], X[i:]).

    Only splits that leave at least min_win_size samples on the left and
    min_win_size + 1 samples on the right are considered, because the MMD
    estimate is unreliable on very small samples. Returns (index, MMD value),
    or (-1, -1) if no split qualifies.
    """
    mmd = lambda X, Y: quadratic_time_mmd(X, Y, kernel)

    maximum = -1
    index = -1

    for i in range(min_win_size, len(X) - min_win_size - 1):
        X0 = X[:i]
        X1 = X[i:]

        result = mmd(X0, X1)

        if (result > maximum):
            maximum = result
            index = i
    return index, maximum


def max_changes(X, m_w_s, gamma, los, alpha=True):
    """Run RMMDCD on the series X with shape (length, dimensions).

    Parameters
    ----------
    X : np.ndarray
        Time series, one row per time step.
    m_w_s : int
        Minimal window size (mws in the thesis).
    gamma : float
        Bandwidth of the Gaussian kernel.
    los : float
        Level of significance in percent, e.g. 55 means alpha = 0.55.
    alpha : bool
        True uses the analytic threshold, False the permutation test.

    Returns the detected change points as indices into X, in the order the
    recursion finds them (not sorted).
    """
    return max_changes_help(X, m_w_s, gamma, los, alpha, 0, 0)


def max_changes_help(X, min_win_size, gamma_i, los, alpha, rank, shift):
    """Recursive step of RMMDCD (thesis Algorithm 2).

    X is the current segment, rank the recursion depth (kept for debugging
    only) and shift the offset of X within the full series, so the returned
    indices refer to the full series.
    """
    kernel = lambda X, Y: gauss_kernel(X, Y, gamma=gamma_i)

    # Stop once the segment is too short to hold two windows of min_win_size
    if (len(X) > min_win_size * 2):

        # Candidate change point: the split with the largest MMD
        index, result = find_max_mmd(X, min_win_size, kernel)

        if (result == -1):
            return []

        # Check if Change is significant

        if (alpha):
            # Analytic threshold (thesis Eq. 3.7, squared because we compare
            # against MMD^2): (K/m + K/n) * (1 + sqrt(2 ln(1/alpha)))^2.
            # The Gaussian kernel is bounded by K = 1.
            upper_bound = ((1 / index) + (1 / (len(X) - index))) * (1 + np.sqrt(2 * np.log((los / 100) ** (-1)))) ** 2
        else:
            # Permutation test: approximate the null distribution of MMD with
            # 400 random permutations of the segment's two halves, then take
            # the (1 - alpha / len(X)) percentile. Dividing alpha by the number
            # of tested splits is the Bonferroni correction (thesis Eq. 3.8).
            mmd = lambda X, Y: quadratic_time_mmd(X, Y, kernel)
            stats = two_sample_permutation_test(mmd, X[:int(len(X) / 2)], X[int(len(X) / 2):], 400)
            upper_bound = np.percentile(stats, ((1 - ((los / 100) / len(X)))) * 100)

        # Not significant: no further change points in this segment
        if (result < upper_bound):
            return []

        #Recursion
        # Search both sides of the detected change point independently
        left = max_changes_help(X[:index], min_win_size, gamma_i, los, alpha, rank + 1, shift)
        right = max_changes_help(X[index:], min_win_size, gamma_i, los, alpha, rank + 1, shift + index)

        return [(index + shift)] + left + right
    else:
        return []

def estimate_gamma(X):
    """Estimate the Gaussian kernel bandwidth (thesis Algorithm 3).

    Computes the median heuristic on 100 random windows of X, each between
    sqrt(len(X)) and len(X) samples long, and returns the smallest value found
    (capped at 1). A small gamma means a wide kernel, which makes RMMDCD find
    large "global" changes before minor local ones.
    """
    sqrt = int(np.sqrt(len(X)))
    min = 1
    for x in range(100):
        # Random window: length between sqrt(len(X)) and len(X), random start
        length = np.random.randint(sqrt, len(X))
        index = np.random.randint(0, len(X) - length)
        g = gaussian_kernel_median_heuristic(X[index:index + length])
        if (g < min):
            min = g

    return min


# --------- Evaluation Metrics by Florian Kalinke ---------
# https://github.com/FlopsKa/StreamDatasets/blob/main/changeds/metrics.py

def true_positives(true_cps, reported_cps, T=5):
    """Count reported change points within T steps of a true change point.

    Each true change point can be matched at most once.
    """
    true_cps = true_cps.copy()
    tps = 0
    for reported_cp in reported_cps:
        for true_cp in true_cps:
            if abs(true_cp - reported_cp) <= T:
                tps += 1
                # Remove the match so it cannot be counted twice
                true_cps.remove(true_cp)
                break
    return tps

def fb_score(true_cps, reported_cps, T=5, beta=1):
    """F-beta score against several annotators (TCPDBench, thesis Eqs. 6.4 and 6.5).

    true_cps is a list with one list of change points per annotator.
    Precision compares the reported points with the union of all annotations;
    recall is computed per annotator and averaged. With beta = 1 this is the F1
    score used in the thesis.
    """
    all_cps = []
    rec = 0
    for x in true_cps:
        # Collect the union of all annotations and sum the per-annotator recall
        all_cps = all_cps + x
        tps = true_positives(x, reported_cps, T=T)
        rec = rec + (tps / len(x))

    rec = rec / len(true_cps)

    tps_all = true_positives(all_cps, reported_cps, T=T)

    prec = tps_all / len(reported_cps)

    if prec == 0:
        return np.nan
    return (1 + beta ** 2) * (prec * rec) / ((beta ** 2 * prec) + rec)


# ---------- Callable functions ----------

def calculate_f1(file, gamma_factor, mws, los, alpha, data_dir=".", annotations="annotations.json"):
    """Label one TCPD data set with RMMDCD and score it against the human annotations.

    Parameters
    ----------
    file : str
        Data set name without extension, e.g. "nile" for nile.json.
    gamma_factor : float
        Factor applied to the estimated kernel bandwidth.
    mws : int
        Minimal window size.
    los : float
        Level of significance in percent.
    alpha : bool
        True for the analytic threshold, False for the permutation test.
    data_dir : str
        Directory containing the TCPD <name>.json files.
    annotations : str
        Path to the TCPDBench annotations.json.

    Returns (F1 score, detected change points including the leading 0).
    """
    data = json.load(open(os.path.join(data_dir, file + ".json")))
    keyword = file

    # TCPD stores one entry per dimension in data["series"], each with a "raw"
    # list of values. Build an array of shape (length, dimensions).
    array = np.asarray(pd.DataFrame(data["series"])["raw"])
    if (len(array) > 1):
        # Multivariate: transpose the per-dimension lists into rows per time step
        array = np.asarray(list(zip(*array)))
    else:
        # Univariate: a single column
        array = np.asarray(pd.DataFrame(data["series"])["raw"][0])[:, np.newaxis]

    gamma = estimate_gamma(array) * gamma_factor
    result = max_changes(array, mws, gamma, los, alpha)

    # TCPDBench counts index 0 as a change point in every annotation, so the
    # prediction includes it too. This keeps precision and recall defined
    # when a series has no other change points.
    result = [0] + result

    # annotations.json maps each data set to {annotator id: [change points]}
    raw = json.load(open(annotations))[keyword]

    comb_list = []

    for x in raw:
        comb_list = comb_list + [[0] + raw[str(x)]]

    burgf1_score = fb_score(comb_list, result, T=5, beta=1)

    return burgf1_score, result


def calculate_f1_all(data_dir=".", annotations="annotations.json", out="results.csv"):
    """Grid search over the RMMDCD parameters on every data set in data_dir.

    Writes one CSV row per (parameter combination, data set) with the columns
    "gamma_factor mws los threshold", data set name, F1 score and detected
    change points, and prints the mean F1 score of each combination.
    """
    # Parameter grid from Table 5.1 of the thesis
    possible_gamma_factors = [0.001, 0.0018, 0.0055, 0.01, 0.018, 0.055, 0.1, 0.18, 0.55, 1, 1.8, 5.5, 10, 55, 100]
    possible_mws = [20,30,40,50]
    possible_los = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 55, 60, 70, 75]
    # True = analytic threshold, False = permutation test
    possible_threshold = [True, False]

    f = open(out, "w", newline="")
    writer = csv.writer(f)

    # Every combination of (gamma factor, mws, level of significance, threshold method)
    possibles = [(a, b, c, d) for a in possible_gamma_factors for b in possible_mws for c in possible_los for d in possible_threshold]

    for (a, b, c, d) in possibles:
        print(str(a) + " " + str(b) + " " + str(c) + str(d))

        # Sum of F1 scores over all data sets for this combination
        a_burg = 0

        # Every .json file in data_dir except the annotations is a data set
        json_files = [pos_json for pos_json in os.listdir(data_dir) if pos_json.endswith('.json')]
        if os.path.basename(annotations) in json_files:
            json_files.remove(os.path.basename(annotations))
        for x in json_files:
            print(x[:-5])
            burg, cps = calculate_f1(x[:-5], a, b, c, d, data_dir, annotations)
            writer.writerow([str(a) + " " + str(b) + " " + str(c) + " " + str(d), x[:-5], burg, cps])
            print("Burg: " + str(burg) + "\n")
            a_burg += burg

        # Mean F1 score of this parameter combination
        a_burg = a_burg / len(json_files)
        print(a_burg)

    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RMMDCD grid search on the TCPDBench data sets.")
    parser.add_argument("--data-dir", default=".", help="directory containing the TCPD <name>.json files")
    parser.add_argument("--annotations", default="annotations.json", help="path to the TCPD annotations.json")
    parser.add_argument("--out", default="results.csv", help="CSV file the per-dataset results are written to")
    args = parser.parse_args()
    calculate_f1_all(args.data_dir, args.annotations, args.out)
