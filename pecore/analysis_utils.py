from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, f1_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .enums import EvalModeEnum


def get_scat_splits(
    df: pd.DataFrame,
    print_stats: bool = True,
    target_column: str = "is_context_sensitive",
    eval_mode: EvalModeEnum = "cti",
) -> pd.DataFrame:
    curr_df = df.copy()
    scat_cs = curr_df["example_idx"] < 250
    scat_ci = curr_df["example_idx"] >= 250
    scat_ok = curr_df["is_example_correct"] == 1
    scat_bad = curr_df["is_example_correct"] == 0
    scat_cs_ok = (curr_df["example_idx"] < 250) & (curr_df["is_example_correct"] == 1)
    scat_ci_ok = (curr_df["example_idx"] >= 250) & (curr_df["is_example_correct"] == 1)
    scat_cs_bad = (curr_df["example_idx"] < 250) & (curr_df["is_example_correct"] == 0)
    scat_ci_bad = (curr_df["example_idx"] >= 250) & (curr_df["is_example_correct"] == 0)
    scat_all = curr_df["example_idx"] >= 0
    splits = {
        "scat_cs_ok": {"test": scat_cs_ok, "train": scat_cs},
        "scat_cs_bad": {"test": scat_cs_bad, "train": scat_cs},
        "scat_cs_all": {"test": scat_cs, "train": scat_cs},
        "scat_ci_ok": {"test": scat_ci_ok, "train": scat_ci},
        "scat_ci_bad": {"test": scat_ci_bad, "train": scat_ci},
        "scat_ci_all": {"test": scat_ci, "train": scat_ci},
        "scat_ok": {"test": scat_ok, "train": scat_all},
        "scat_bad": {"test": scat_bad, "train": scat_all},
        "scat_all": {"test": scat_all, "train": scat_all},
    }
    if print_stats:
        for split_name, split in splits.items():
            for fold in ["train", "test"]:
                print(f"{split_name:<10s} {fold}:\t{str(split[fold].sum()):<4s} ex", end=",\t")
                print(f"{target_column}: {curr_df[split[fold]][target_column].sum()} ex", end=",\t")
                print(
                    "is_example_correct:"
                    f" {curr_df[split[fold]].groupby('example_idx').mean(numeric_only=True).is_example_correct.sum()} vals"
                )
    if eval_mode == EvalModeEnum.CCI:
        filtered_splits = {}
        for split_name, split in splits.items():
            if curr_df[split["test"]][target_column].sum() != 0:
                filtered_splits[split_name] = split
        return filtered_splits
    return splits


def get_cti_mix_features(n_layers: int, has_target_context: bool) -> List[str]:
    cti_mix = [
        "kl_divergence",
        "src_ctx_attr",
        "contrast_prob_diff",
        "contrast_prob",
        "src_curr_attr",
    ] + [f"kl_div_l{i}" for i in range(n_layers)]
    if has_target_context:
        cti_mix += ["tgt_ctx_attr"]
    return cti_mix


def get_metrics_result(
    df: pd.DataFrame,
    train_mask: pd.Series,
    test_mask: pd.Series,
    scores_columns: List[str] = None,
    do_random: bool = False,
    target_column: str = "is_context_sensitive",
    n_cv_splits: int = 10,
    seed: int = 42,
    fillna: bool = True,
) -> Dict[str, float]:
    if fillna:
        df = df.fillna(df.mean(numeric_only=True))

    # Features and target for the updated DataFrame
    X = df[train_mask][scores_columns]
    y = df[train_mask][target_column]
    X_split = df[test_mask][scores_columns]

    # Indices of the split, later used to filter CV test subsets for split-only performances
    split_ix = np.where(X.index.isin(X_split.index))[0]

    X = X.to_numpy()
    y = y.to_numpy()

    cv = StratifiedKFold(n_splits=n_cv_splits, shuffle=True, random_state=seed)

    # Train the classifier
    clf = RandomForestClassifier(random_state=seed, class_weight="balanced")
    dummy = DummyClassifier(strategy="stratified", random_state=seed)

    f1_scores = []
    auprc_scores = []

    for train_ix, test_ix in cv.split(X, y):
        # Make sure to filter only test examples matching split condition, i.e. filtering the test split for every
        # CV fold based on the test_mask condition. If trainand test masks coincide, this is equivalent to regular CV.
        split_test_ix = np.intersect1d(test_ix, split_ix)
        X_train, X_test = X[train_ix, :], X[split_test_ix, :]
        y_train, y_test = y[train_ix], y[split_test_ix]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        if not do_random:
            clf.fit(X_train_scaled, y_train)

            # predict on the test set
            y_pred = clf.predict(X_test_scaled)

            # get feature importances
            # feature_importances.append(clf.coef_)
            predicted_proba = clf.predict_proba(X_test_scaled)[:, 1]
        else:
            dummy.fit(X_train_scaled, y_train)
            y_pred = dummy.predict(X_test_scaled)
            predicted_proba = dummy.predict_proba(X_test_scaled)[:, 1]

        # get classification report
        f1 = f1_score(y_test, y_pred, average="macro")
        precision, recall, _ = precision_recall_curve(y_test, predicted_proba)
        auprc = auc(recall, precision)
        f1_scores.append(f1)
        auprc_scores.append(auprc)

    # Average classification report across folds
    return {
        "avg_macro_f1": np.mean(f1_scores).round(4),
        "std_macro_f1": np.std(f1_scores).round(4),
        "avg_auprc": np.mean(auprc_scores).round(4),
        "std_auprc": np.std(auprc_scores).round(4),
    }