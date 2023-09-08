from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, f1_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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


def get_metrics_result_with_trained_model(
    df: pd.DataFrame,
    train_mask: pd.Series,
    test_mask: pd.Series,
    scores_columns: List[str],
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


def get_metric_results_from_scores(
    df: pd.DataFrame,
    test_mask: pd.Series,
    score_column: str,
    do_random: bool = False,
    target_column: str = "is_context_sensitive",
    example_id_column: str = "example_idx",
    cti_id_column: str = "cti_idx",
    average_example_scores: bool = True,
    pos_column: str = "pos",
    valid_pos: List[str] = None,
    input_type_column: str = "side",
    valid_input_types: List[str] = ["S", "T"],
    initial_only: bool = False,
    token_column: str = "token",
    initial_char: str = "▁",
    fillna: bool = True,
    std_threshold: float = 1.0,
    n_random_matches_per_example: int = 1,
) -> Tuple[Dict[str, float], List[bool]]:
    if fillna:
        df = df.fillna(df.mean(numeric_only=True))
    df = df[test_mask]
    if input_type_column in df.columns:
        df = df[df[input_type_column].isin(valid_input_types)]
    scaler = MinMaxScaler()
    if average_example_scores:
        examples = df[example_id_column].unique()
        auprc_scores = []
        f1_scores = []
        all_preds = []
        for example in examples:
            df_ex = df[df[example_id_column] == example]
            ex_tgt = df_ex[target_column]

            if valid_pos is not None:
                c_pos = list(df_ex[pos_column])
                ex_tgt = [s if c_pos[i] in valid_pos else 0 for i, s in enumerate(ex_tgt)]
            if initial_only:
                c_tok = list(df_ex[token_column])
                ex_tgt = [s if initial_char in c_tok[i] else 0 for i, s in enumerate(ex_tgt)]
            if do_random:
                ex_scores = np.random.rand(len(ex_tgt))
                threshold = ex_scores.mean() + (std_threshold * ex_scores.std())
                ex_scores = ex_scores * threshold
                ex_matches_pos = np.random.choice(len(ex_tgt), n_random_matches_per_example, replace=False)
                ex_scores[ex_matches_pos] += threshold
            else:
                ex_scores = df_ex[score_column].to_numpy()
            # Select only scores one standard deviation away from the mean
            threshold = ex_scores.mean() + (std_threshold * ex_scores.std())
            ex_scores_binary = ex_scores > threshold
            if cti_id_column in df_ex.columns:
                # If multiple target indices are available for the same sequence, we take the max score for the metric
                # and set prediction to True if at least one of the target indices is predicted as True
                n_cti_idxs = len(df_ex[cti_id_column].unique())
                if n_cti_idxs > 1:
                    ex_scores_binary = ex_scores_binary.reshape(-1, n_cti_idxs).mean(axis=1) > 0
                    ex_scores = ex_scores.reshape(-1, n_cti_idxs).max(axis=1)
                    ex_tgt = ex_tgt[: len(ex_tgt) // n_cti_idxs]

            ex_scores = scaler.fit_transform(ex_scores.reshape(-1, 1))
            precision, recall, _ = precision_recall_curve(ex_tgt, ex_scores)
            auprc = auc(recall, precision)
            f1_val = f1_score(ex_tgt, ex_scores_binary, average="macro")
            auprc_scores.append(auprc)
            f1_scores.append(f1_val)
            all_preds += ex_scores_binary.reshape(1, -1).squeeze().tolist()
        return {
            "avg_auprc": np.mean(auprc_scores).round(4),
            "std_auprc": np.std(auprc_scores).round(4),
            "avg_macro_f1": np.mean(f1_scores).round(4),
            "std_macro_f1": np.std(f1_scores).round(4),
        }, all_preds
    else:
        tgt = df[target_column]
        if valid_pos is not None:
            tgt = [s if df[pos_column][i] in valid_pos else 0 for i, s in enumerate(tgt)]
        if initial_only:
            tgt = [s if initial_char in df[token_column][i] else 0 for i, s in enumerate(tgt)]
        if do_random:
            scores = np.random.rand(len(tgt)) * 0.5
            matches_pos = np.random.choice(len(tgt), n_random_matches_per_example, replace=False)
            scores[matches_pos] += 0.5
        else:
            scores = df[score_column].to_numpy()
        # Select only scores one standard deviation away from the mean
        scores_binary = scores > (scores.mean() + (std_threshold * scores.std()))
        scores = scaler.fit_transform(scores.reshape(-1, 1))
        precision, recall, _ = precision_recall_curve(tgt, scores)
        auprc = auc(recall, precision)
        f1_val = f1_score(tgt, scores_binary, average="macro")
        return {
            "auprc": auprc.round(4),
            "macro_f1": f1_val.round(4),
        }, scores_binary.tolist()
