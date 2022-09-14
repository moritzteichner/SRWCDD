# Imports

import numpy as np
from scipy.spatial.distance import squareform, pdist, cdist
import json
import pandas as pd

import sys

import os

import csv

import itertools

from matplotlib import pyplot as plt


# --------- Underlying functionality ---------

# ------ Copied from https://github.com/karlnapf/ds3_kernel_testing/blob/master/solutions.ipynb ------

def sq_distances(X, Y=None):
    assert (X.ndim == 2)

    if Y is None:
        sq_dists = squareform(pdist(X, 'sqeuclidean'))
    else:
        assert (Y.ndim == 2)
        assert (X.shape[1] == Y.shape[1])
        sq_dists = cdist(X, Y, 'sqeuclidean')

    return sq_dists


def gauss_kernel(X, Y=None, gamma=1.0):
    sq_dists = sq_distances(X, Y)
    K = np.exp(-gamma * sq_dists)
    return K


def quadratic_time_mmd(X, Y, kernel):
    assert X.ndim == Y.ndim == 2
    K_XX = kernel(X, X)
    K_XY = kernel(X, Y)
    K_YY = kernel(Y, Y)

    n = len(K_XX)
    m = len(K_YY)

    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)
    mmd = np.sum(K_XX) / (n * (n - 1)) + np.sum(K_YY) / (m * (m - 1)) - 2 * np.sum(K_XY) / (n * m)
    return mmd


def gaussian_kernel_median_heuristic(Z):
    sq_dists = sq_distances(Z)
    np.fill_diagonal(sq_dists, np.nan)
    sq_dists = np.ravel(sq_dists)
    sq_dists = sq_dists[~np.isnan(sq_dists)]
    median_dist = np.median(np.sqrt(sq_dists))
    return (1 / median_dist)


def two_sample_permutation_test(test_statistic, X, Y, num_permutations, prog_bar=True):
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

        perm_inds = np.random.permutation(len(Z))
        Z = Z[perm_inds]
        X_ = Z[:len(X)]
        Y_ = Z[len(X):]
        my_test_statistic = test_statistic(X_, Y_)
        statistics[i] = my_test_statistic
    return statistics


# ------ Copied End------

def find_max_mmd(X, min_win_size, kernel):
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
    return max_changes_help(X, m_w_s, gamma, los, alpha, 0, 0)


def max_changes_help(X, min_win_size, gamma_i, los, alpha, rank, shift):
    kernel = lambda X, Y: gauss_kernel(X, Y, gamma=gamma_i)

    if (len(X) > min_win_size * 2):

        index, result = find_max_mmd(X, min_win_size, kernel)

        if (result == -1):
            return []

        # Check if Change is significant

        if (alpha):
            upper_bound = ((1 / index) + (1 / (len(X) - index))) * (1 + np.sqrt(2 * np.log((los / 100) ** (-1)))) ** 2
        else:
            mmd = lambda X, Y: quadratic_time_mmd(X[:, np.newaxis], Y[:, np.newaxis], kernel)
            stats = two_sample_permutation_test(mmd, X[:int(len(X) / 2)], X[int(len(X) / 2):], 400)
            upper_bound = np.percentile(stats, ((1 - ((los / 100) / len(X)))) * 100)

        if (result < upper_bound):
            return []

        #Recursion
        left = max_changes_help(X[:index], min_win_size, gamma_i, los, rank + 1, shift)
        right = max_changes_help(X[index:], min_win_size, gamma_i, los, rank + 1, shift + index)

        return [(index + shift)] + left + right
    else:
        return []

def estimate_gamma(X):
    sqrt = int(np.sqrt(len(X)))
    min = 1
    for x in range(100):
        length = np.random.randint(sqrt, len(X))
        index = np.random.randint(0, len(X) - length)
        g = gaussian_kernel_median_heuristic(X[index:index + length])
        if (g < min):
            min = g

    return min


# --------- Evaluation Metrics by Florian Kalinke ---------
# https://github.com/FlopsKa/StreamDatasets/blob/main/changeds/metrics.py

def true_positives(true_cps, reported_cps, T=5):
    true_cps = true_cps.copy()
    tps = 0
    for reported_cp in reported_cps:
        for true_cp in true_cps:
            if abs(true_cp - reported_cp) <= T:
                tps += 1
                true_cps.remove(true_cp)
                break
    return tps

def fb_score(true_cps, reported_cps, T=5, beta=1):
    all_cps = []
    rec = 0
    for x in true_cps:
        all_cps = all_cps + x
        tps = true_positives(x, reported_cps, T=5)
        rec = rec + (tps / len(x))

    rec = rec / len(true_cps)

    tps_all = true_positives(all_cps, reported_cps, T=5)

    prec = tps_all / len(reported_cps)

    if prec == 0:
        return np.nan
    return (1 + beta ** 2) * (prec * rec) / ((beta ** 2 * prec) + rec)


# ---------- Callable functions ----------

def calculate_f1(file, gamma_factor, mws, los, alpha):
    data = json.load(open(file + ".json"))
    keyword = file

    array = np.asarray(pd.DataFrame(data["series"])["raw"])
    if (len(array) > 1):
        array = np.asarray(list(zip(*array)))
    else:
        array = np.asarray(pd.DataFrame(data["series"])["raw"][0])[:, np.newaxis]

    gamma = estimate_gamma(array) * gamma_factor
    result = max_changes(array, mws, gamma, los, alpha)

    result = [0] + result

    raw = json.load(open('annotations.json'))[keyword]

    comb_list = []

    for x in raw:
        comb_list = comb_list + [[0] + raw[str(x)]]

    burgf1_score = fb_score(comb_list, result, T=5, beta=1)

    return burgf1_score, result


def calculate_f1_all():
    possible_gamma_factors = [1/1000, 1/100, 5/100, 1/10, 0.5, 1, 5, 10]
    possible_mws = [20,30,40,50]
    possible_los = [2,5,20,40,50,55,60,65]
    possible_threshold = [True, False]

    f = open("results", "w")
    writer = csv.writer(f)

    possibles = [(a, b, c, d) for a in possible_gamma_factors for b in possible_mws for c in possible_los for d in possible_threshold]

    for (a, b, c, d) in possibles:
        print(str(a) + " " + str(b) + " " + str(c) + str(d))

        a_burg = 0

        json_files = [pos_json for pos_json in os.listdir() if pos_json.endswith('.json')]
        json_files.remove("annotations.json")
        for x in json_files:
            print(x[:-5])
            burg, cps = calculate_f1(x[:-5], a, b, c, d)
            writer.writerow([str(a) + " " + str(b) + " " + str(c) + " " + str(d), x[:-5], burg, cps])
            print("Burg: " + str(burg) + "\n")
            a_burg += burg

        a_burg = a_burg / len(json_files)
        print(a_burg)

    f.close()
