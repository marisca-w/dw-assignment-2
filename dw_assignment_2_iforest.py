"""
Isolation Forest replication: Section 5.1 (Table 3) of Liu, Ting and Zhou (2008)
================================================================================


How to run
----------
1. Make sure to do the following:
    pip install ucimlrepo scikit-learn pandas numpy
2. Run the whole file, or run it cell by cell.
3. Output: tables in the console and a file called results_5_1.csv.

"""

# %% 1. Imports
import time

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype, fetch_kddcup99
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from ucimlrepo import fetch_ucirepo


# %% 2. Settings
# iForest defaults
N_TREES = 100
SUBSAMPLE = 256

# Experiment design
SEEDS = range(10)          
SETUPS = ["split"]         
TEST_FRACTION = 0.5        
N_JOBS = 1               

# Baselines
K_LOF = 10                 
K_KNN = 5                  
N_TREES_RF = 100          

# Which datasets each baseline is run on. These match the datasets where the paper reports a result for that method
LOF_DATASETS = {"Breastw", "Ionosphere", "Satellite", "Shuttle"}
KNN_DATASETS = {"Breastw", "Ionosphere", "Satellite", "Shuttle"}
RF_DATASETS = {"Breastw", "Ionosphere"}  # the paper reports RF only on small datasets

METHODS = ["iForest", "LOF", "kNN-distance", "RF"]


# %% 3. Reference values from the paper
# Table 2: (instances, attributes)
PAPER_SHAPE = {
    "Http": (567497, 3), "ForestCover": (286048, 10), "Shuttle": (49097, 9),
    "Satellite": (6435, 36), "Breastw": (683, 9), "Ionosphere": (351, 32),
}

# Table 3. AUC of each method, iForest training/evaluation/total time, and the total times 
PAPER = {
    "Http":        dict(iforest=1.00, lof=None, orca=0.36, rf=None,
                        train=0.25, eval=15.33, itotal=15.58,
                        orca_s=9487.47, lof_s=None, rf_s=None),
    "ForestCover": dict(iforest=0.88, lof=None, orca=0.83, rf=None,
                        train=0.76, eval=15.57, itotal=16.33,
                        orca_s=6995.17, lof_s=None, rf_s=None),
    "Shuttle":     dict(iforest=1.00, lof=0.55, orca=0.60, rf=None,
                        train=0.30, eval=2.83, itotal=3.13,
                        orca_s=156.66, lof_s=7489.74, rf_s=None),
    "Satellite":   dict(iforest=0.71, lof=0.52, orca=0.65, rf=None,
                        train=0.46, eval=1.17, itotal=1.63,
                        orca_s=8.51, lof_s=217.39, rf_s=None),
    "Breastw":     dict(iforest=0.99, lof=0.37, orca=0.98, rf=0.97,
                        train=0.17, eval=0.11, itotal=0.28,
                        orca_s=0.04, lof_s=1.77, rf_s=3.10),
    "Ionosphere":  dict(iforest=0.85, lof=0.89, orca=0.92, rf=0.85,
                        train=0.33, eval=0.15, itotal=0.48,
                        orca_s=0.04, lof_s=0.64, rf_s=0.83),
}


# %% 4. Data loaders
# Each loader returns X (instances x attributes, floats) and y (1 = anomaly, 0 = normal).

def load_http():
    """KDD Cup 99, http subset, anomaly = any attack. Large download on first use."""
    X, y = fetch_kddcup99(subset="http", percent10=False, return_X_y=True)
    is_normal = pd.Series(y).astype(str).str.contains("normal").to_numpy()
    return np.asarray(X, dtype=float), (~is_normal).astype(int)


def load_forestcover():
    """Forest cover type: class 4 is the anomaly, class 2 the normal class,
    and only the 10 continuous attributes (the first 10 columns) are kept."""
    X, y = fetch_covtype(return_X_y=True)
    keep = np.isin(y, [2, 4])
    X, y = X[keep][:, :10], y[keep]
    return X.astype(float), (y == 4).astype(int)


def load_shuttle():
    """Statlog Shuttle: class 1 is normal, classes 2, 3, 5, 6, 7 are anomalies and
    class 4 is dropped. The UCI copy has only 7 of the paper's 9 attributes."""
    d = fetch_ucirepo(id=148)
    X, y = d.data.features, d.data.targets.iloc[:, 0].astype(int)
    keep = y != 4
    return X[keep].to_numpy(float), (y[keep] != 1).astype(int).to_numpy()


def load_satellite():
    """Statlog Landsat Satellite: the three smallest classes (2, 4, 5) are anomalies."""
    d = fetch_ucirepo(id=146)
    X, y = d.data.features, d.data.targets.iloc[:, 0].astype(int)
    return X.to_numpy(float), y.isin([2, 4, 5]).astype(int).to_numpy()


def load_breastw():
    """Breast Cancer Wisconsin: malignant (class 4) is the anomaly; rows with a
    missing value are dropped."""
    d = fetch_ucirepo(id=15)
    X, y = d.data.features, d.data.targets.iloc[:, 0]
    keep = X.notna().all(axis=1)
    X, y = X[keep], y[keep]
    return X.to_numpy(float), (y == 4).astype(int).to_numpy()


def load_ionosphere():
    """Ionosphere: 'bad' is the anomaly; binary and constant attributes are removed
    because the paper keeps only continuous ones."""
    d = fetch_ucirepo(id=52)
    X, y = d.data.features, d.data.targets.iloc[:, 0]
    X = X.loc[:, X.nunique() > 2]
    return X.to_numpy(float), (y == "b").astype(int).to_numpy()


# Datasets in the order of the paper's tables (largest first)
LOADERS = {
    "Http": load_http, "ForestCover": load_forestcover, "Shuttle": load_shuttle,
    "Satellite": load_satellite, "Breastw": load_breastw, "Ionosphere": load_ionosphere,
}


# %% 5. Methods
# Every method returns (AUC, training seconds, evaluation seconds)

def get_parts(X, y, setup, seed):
    """Return the training data, the test data and the test labels for one run."""
    if setup == "full":                      # fit and score on all the data
        return X, X, y
    X_train, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION, random_state=seed)
    return X_train, X_test, y_test


def run_iforest(X_train, X_test, y_test, seed):
    model = IsolationForest(n_estimators=N_TREES, max_samples=SUBSAMPLE,
                            random_state=seed, n_jobs=N_JOBS)
    start = time.perf_counter()
    model.fit(X_train)
    train_seconds = time.perf_counter() - start

    start = time.perf_counter()
    scores = -model.score_samples(X_test)    # higher = more anomalous
    eval_seconds = time.perf_counter() - start
    return roc_auc_score(y_test, scores), train_seconds, eval_seconds


def run_lof(X_train, X_test, y_test, setup):
    if setup == "full":
        lof = LocalOutlierFactor(n_neighbors=K_LOF)
        lof.fit_predict(X_test)              # fitting and scoring happen in one step
        scores = -lof.negative_outlier_factor_
        return roc_auc_score(y_test, scores), np.nan, np.nan

    lof = LocalOutlierFactor(n_neighbors=K_LOF, novelty=True)
    start = time.perf_counter()
    lof.fit(X_train)
    train_seconds = time.perf_counter() - start

    start = time.perf_counter()
    scores = -lof.score_samples(X_test)      # higher = more anomalous
    eval_seconds = time.perf_counter() - start
    return roc_auc_score(y_test, scores), train_seconds, eval_seconds


def run_knn_distance(X_train, X_test, y_test, setup):
    start = time.perf_counter()
    neighbours = NearestNeighbors(n_neighbors=K_KNN, n_jobs=N_JOBS).fit(X_train)
    train_seconds = time.perf_counter() - start

    start = time.perf_counter()
    if setup == "full":
        distances, _ = neighbours.kneighbors()        # a point is not its own neighbour
    else:
        distances, _ = neighbours.kneighbors(X_test)
    eval_seconds = time.perf_counter() - start
    return roc_auc_score(y_test, distances[:, -1]), train_seconds, eval_seconds


def run_rf_proximity(X_train, X_test, y_test, seed, setup):
    """Unsupervised Random Forests (Shi and Horvath): tell real data from synthetic data,
    then call an instance anomalous if it has low proximity to the real instances.
    Proximity of two instances = share of trees in which they share a leaf, so the
    average proximity of one instance to all training instances can be computed from
    the number of training instances in its leaf, without building an n x n matrix."""
    rng = np.random.default_rng(seed)
    n_train = len(X_train)

    start = time.perf_counter()
    synthetic = rng.uniform(X_train.min(axis=0), X_train.max(axis=0), size=X_train.shape)
    forest = RandomForestClassifier(n_estimators=N_TREES_RF, random_state=seed,
                                    n_jobs=N_JOBS)
    forest.fit(np.vstack([X_train, synthetic]),
               np.r_[np.zeros(n_train), np.ones(n_train)])   # 0 = real, 1 = synthetic
    train_leaves = forest.apply(X_train)                      # (instances, trees)
    train_seconds = time.perf_counter() - start

    start = time.perf_counter()
    test_leaves = forest.apply(X_test)
    shared = np.zeros(len(X_test))   # summed over trees: training instances sharing a leaf
    for t, tree in enumerate(forest.estimators_):
        counts = np.bincount(train_leaves[:, t], minlength=tree.tree_.node_count)
        shared += counts[test_leaves[:, t]]
    shared /= forest.n_estimators                             # average over trees
    if setup == "full":              # the instance itself is among the training instances
        mean_proximity = (shared - 1) / (n_train - 1)
    else:
        mean_proximity = shared / n_train
    eval_seconds = time.perf_counter() - start
    return roc_auc_score(y_test, -mean_proximity), train_seconds, eval_seconds


def run_method(method, X, y, setup, seed):
    """Run one method once; return (AUC, train seconds, eval seconds, total seconds)."""
    X_train, X_test, y_test = get_parts(X, y, setup, seed)
    start = time.perf_counter()
    if method == "iForest":
        auc, train_s, eval_s = run_iforest(X_train, X_test, y_test, seed)
    elif method == "LOF":
        auc, train_s, eval_s = run_lof(X_train, X_test, y_test, setup)
    elif method == "kNN-distance":
        auc, train_s, eval_s = run_knn_distance(X_train, X_test, y_test, setup)
    else:
        auc, train_s, eval_s = run_rf_proximity(X_train, X_test, y_test, seed, setup)
    return auc, train_s, eval_s, time.perf_counter() - start


def applies(method, dataset):
    """Is this method run on this dataset?"""
    if method == "LOF":
        return dataset in LOF_DATASETS
    if method == "kNN-distance":
        return dataset in KNN_DATASETS
    if method == "RF":
        return dataset in RF_DATASETS
    return True


# %% 6. Experiment
def nan_if_none(value):
    return np.nan if value is None else value


def evaluate_dataset(name, X, y):
    """Run every applicable method on one dataset; return one result row per method."""
    paper = PAPER[name]
    paper_auc = {"iForest": paper["iforest"], "LOF": paper["lof"],
                 "kNN-distance": None, "RF": paper["rf"]}
    paper_total = {"iForest": paper["itotal"], "LOF": paper["lof_s"],
                   "kNN-distance": None, "RF": paper["rf_s"]}
    rows = []
    for setup in SETUPS:
        for method in METHODS:
            if not applies(method, name):
                continue
            # LOF and kNN have no randomness when scoring all the data: one run is enough
            deterministic = method in ("LOF", "kNN-distance") and setup == "full"
            seeds = [0] if deterministic else list(SEEDS)

            start = time.time()
            runs = np.array([run_method(method, X, y, setup, s) for s in seeds])
            auc, train_s, eval_s, total_s = runs.T
            print(f"   {method:<13} ({setup}): AUC {auc.mean():.3f} +/- {auc.std():.3f}"
                  f"   [{time.time() - start:.0f}s]")

            is_iforest = method == "iForest"
            rows.append({
                "dataset": name, "method": method, "setup": setup, "runs": len(seeds),
                "auc_mean": auc.mean(), "auc_std": auc.std(),
                "paper_auc": nan_if_none(paper_auc[method]),
                "train_s": train_s.mean(), "eval_s": eval_s.mean(),
                "total_s": total_s.mean(),
                "paper_train_s": paper["train"] if is_iforest else np.nan,
                "paper_eval_s": paper["eval"] if is_iforest else np.nan,
                "paper_total_s": nan_if_none(paper_total[method]),
                "paper_orca_auc": nan_if_none(paper["orca"]),
                "paper_orca_total_s": nan_if_none(paper["orca_s"]),
            })
    return rows


# %% 7. Results tables
def fmt(value, digits=2):
    return "-" if value is None or pd.isna(value) else f"{value:.{digits}f}"


def lookup(results, dataset, method, setup):
    hit = results[(results.dataset == dataset) & (results.method == method)
                  & (results.setup == setup)]
    return None if hit.empty else hit.iloc[0]


def auc_table(results, setup):
    """One row per dataset: our AUC (mean +/- std over runs) next to the paper's."""
    def mine(dataset, method):
        row = lookup(results, dataset, method, setup)
        return "-" if row is None else f"{row.auc_mean:.3f} +/- {row.auc_std:.3f}"

    rows = []
    for name in LOADERS:
        p = PAPER[name]
        rows.append({
            "dataset": name,
            "iForest (mine)": mine(name, "iForest"), "iForest (paper)": fmt(p["iforest"]),
            "LOF (mine)": mine(name, "LOF"), "LOF (paper)": fmt(p["lof"]),
            "RF (mine)": mine(name, "RF"), "RF (paper)": fmt(p["rf"]),
            "kNN-dist (mine)": mine(name, "kNN-distance"), "ORCA (paper)": fmt(p["orca"]),
        })
    return pd.DataFrame(rows)


def total_time_table(results, setup):
    """Total seconds per method (training plus evaluation), ours against the paper's."""
    def mine(dataset, method):
        row = lookup(results, dataset, method, setup)
        return "-" if row is None else fmt(row.total_s)

    rows = []
    for name in LOADERS:
        p = PAPER[name]
        rows.append({
            "dataset": name,
            "iForest (mine)": mine(name, "iForest"), "iForest (paper)": fmt(p["itotal"]),
            "LOF (mine)": mine(name, "LOF"), "LOF (paper)": fmt(p["lof_s"]),
            "RF (mine)": mine(name, "RF"), "RF (paper)": fmt(p["rf_s"]),
            "kNN-dist (mine)": mine(name, "kNN-distance"), "ORCA (paper)": fmt(p["orca_s"]),
        })
    return pd.DataFrame(rows)


def iforest_time_table(results, setup):
    """iForest training and evaluation seconds, ours against the paper's."""
    rows = []
    for name in LOADERS:
        r = lookup(results, name, "iForest", setup)
        if r is None:
            continue
        rows.append({
            "dataset": name,
            "train s (mine)": fmt(r.train_s), "train s (paper)": fmt(r.paper_train_s),
            "eval s (mine)": fmt(r.eval_s), "eval s (paper)": fmt(r.paper_eval_s),
        })
    return pd.DataFrame(rows)


def main():
    all_rows = []
    for name, load in LOADERS.items():
        X, y = load()
        print(f"{name}: got {X.shape}, paper {PAPER_SHAPE[name]}, anomalies {y.mean():.2%}")
        all_rows += evaluate_dataset(name, X, y)

    results = pd.DataFrame(all_rows)
    for setup in SETUPS:
        print(f"\n===== AUC, setup = {setup} (mine: mean +/- std over runs) =====")
        print(auc_table(results, setup).to_string(index=False))
        print(f"\n===== Total seconds per method, setup = {setup} =====")
        print(total_time_table(results, setup).to_string(index=False))
        print(f"\n===== iForest training and evaluation seconds, setup = {setup} =====")
        print(iforest_time_table(results, setup).to_string(index=False))

    results.round(4).to_csv("results_5_1.csv", index=False)
    print("\nSaved results_5_1.csv")
    return results


if __name__ == "__main__":
    results = main()