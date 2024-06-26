from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata, sem, t
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, f1_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .enums import EvalModeEnum, TaggedDatasetEnum


def mrr(y, pred) -> Optional[float]:
    rank_descending = [x - 1 for x in rankdata(pred, method="dense")][::-1]
    gold_tags = [i for i, val in enumerate(y) if val == 1 and i in rank_descending]
    if not gold_tags:
        return np.nan
    return 1 / (min(rank_descending.index(tag) for tag in gold_tags) + 1)


def dot(y, pred) -> float:
    return np.matmul(pred, y).item()


def conf_bounds(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sem(a)
    h = se * t.ppf((1 + confidence) / 2.0, n - 1)
    return round(m - h, 4), round(m + h, 4)


def get_splits(
    dataset: TaggedDatasetEnum,
    df: pd.DataFrame,
    target_column: str = "is_context_sensitive",
    eval_mode: EvalModeEnum = "cti",
    print_stats: bool = True,
) -> Dict[str, Dict[str, pd.Series]]:
    if dataset == TaggedDatasetEnum.SCAT:
        return get_scat_splits(df, target_column=target_column, eval_mode=eval_mode, print_stats=print_stats)
    elif dataset == TaggedDatasetEnum.DISC_EVAL_MT:
        return get_disc_eval_mt_splits(df, target_column=target_column, print_stats=print_stats)
    else:
        raise ValueError(f"Invalid dataset {dataset}")


def get_max_idx_for_missing_examples(dataset: TaggedDatasetEnum) -> Optional[int]:
    if dataset == TaggedDatasetEnum.SCAT:
        return 500
    elif dataset == TaggedDatasetEnum.DISC_EVAL_MT:
        return 200
    else:
        return None


def get_scat_splits(
    df: pd.DataFrame,
    print_stats: bool = True,
    target_column: str = "is_context_sensitive",
    eval_mode: EvalModeEnum = "cti",
) -> pd.DataFrame:
    curr_df = df.copy()
    scat_cs = curr_df["example_idx"] < 250
    scat_ci = curr_df["example_idx"] >= 250
    scat_cs_ok = (curr_df["example_idx"] < 250) & (curr_df["is_example_correct"] == 1)
    scat_cs_bad = (curr_df["example_idx"] < 250) & (curr_df["is_example_correct"] == 0)
    scat_cs_flipped = (curr_df["example_idx"] < 250) & (curr_df["is_example_flipped"] == 1)
    scat_cs_stable_ok = (
        (curr_df["example_idx"] < 250) & (curr_df["is_example_flipped"] == 0) & (curr_df["is_example_correct"] == 1)
    )
    scat_cs_stable_bad = (
        (curr_df["example_idx"] < 250) & (curr_df["is_example_flipped"] == 0) & (curr_df["is_example_correct"] == 0)
    )
    splits = {
        "scat_cs_ok": {"test": scat_cs_ok, "train": scat_cs},
        "scat_cs_bad": {"test": scat_cs_bad, "train": scat_cs},
        "scat_cs_all": {"test": scat_cs, "train": scat_cs},
        "scat_ci_all": {"test": scat_ci, "train": scat_ci},
        "scat_cs_flipped": {"test": scat_cs_flipped, "train": scat_cs},
        "scat_cs_stable_ok": {"test": scat_cs_stable_ok, "train": scat_cs},
        "scat_cs_stable_bad": {"test": scat_cs_stable_bad, "train": scat_cs},
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


def get_disc_eval_mt_splits(
    df: pd.DataFrame,
    print_stats: bool = True,
    target_column: str = "is_context_sensitive",
) -> pd.DataFrame:
    curr_df = df.copy()
    disc_eval_mt_ok = curr_df["is_example_correct"] == 1
    disc_eval_mt_bad = curr_df["is_example_correct"] == 0
    disc_eval_mt_flipped = curr_df["is_example_flipped"] == 1
    disc_eval_mt_stable_ok = (curr_df["is_example_flipped"] == 0) & (curr_df["is_example_correct"] == 1)
    disc_eval_mt_stable_bad = (curr_df["is_example_flipped"] == 0) & (curr_df["is_example_correct"] == 0)
    disc_eval_mt_all = curr_df["example_idx"] >= 0
    splits = {
        "disc_eval_mt_ok": {"test": disc_eval_mt_ok, "train": disc_eval_mt_all},
        "disc_eval_mt_bad": {"test": disc_eval_mt_bad, "train": disc_eval_mt_all},
        "disc_eval_mt_all": {"test": disc_eval_mt_all, "train": disc_eval_mt_all},
        "disc_eval_mt_flipped": {"test": disc_eval_mt_flipped, "train": disc_eval_mt_all},
        "disc_eval_mt_stable_ok": {"test": disc_eval_mt_stable_ok, "train": disc_eval_mt_all},
        "disc_eval_mt_stable_bad": {"test": disc_eval_mt_stable_bad, "train": disc_eval_mt_all},
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
    special_tokens_to_remove: List[str] = None,
    max_idx_for_missing_examples: Optional[int] = None,
) -> Tuple[Dict[str, float], List[bool]]:
    if fillna:
        df = df.fillna(df.mean(numeric_only=True))
    df = df[test_mask]
    if input_type_column in df.columns:
        df = df[df[input_type_column].isin(valid_input_types)]
    scaler = MinMaxScaler()
    if average_example_scores:
        available_examples = df[example_id_column].unique()
        if max_idx_for_missing_examples:
            examples = list(range(max_idx_for_missing_examples))
        else:
            examples = available_examples
        auprc_scores = []
        f1_scores = []
        all_preds = []
        mrr_scores = []
        dot_scores = []
        for example in examples:
            if example not in available_examples:
                # Only MRR takes into account examples that weren't computed because an attribution target wasn't
                # identified in the previous step - should be used as preferred metric for CCI
                # mrr_scores.append(0)
                continue
            df_ex = df[df[example_id_column] == example]
            ex_tgt = df_ex[target_column]
            c_tok = list(df_ex[token_column])

            if valid_pos is not None:
                c_pos = list(df_ex[pos_column])
                ex_tgt = [s if c_pos[i] in valid_pos else 0 for i, s in enumerate(ex_tgt)]
            if initial_only:
                ex_tgt = [s if initial_char in c_tok[i] else 0 for i, s in enumerate(ex_tgt)]
            if do_random:
                np.random.seed(42)
                ex_scores = np.random.rand(len(ex_tgt))
                threshold = ex_scores.mean() + (std_threshold * ex_scores.std())
                ex_scores = ex_scores * threshold
                ex_matches_pos = np.random.choice(len(ex_tgt), n_random_matches_per_example, replace=False)
                ex_scores[ex_matches_pos] += threshold
            else:
                ex_scores = df_ex[score_column].to_numpy()
            if special_tokens_to_remove is not None:
                ex_scores_squeezed = ex_scores.squeeze()
                ex_scores_without_special_tokens = [
                    s for i, s in enumerate(ex_scores_squeezed) if c_tok[i] not in special_tokens_to_remove
                ]
                ex_scores_special_tokens_mask = np.array(
                    [True if c_tok[i] in special_tokens_to_remove else False for i in range(len(c_tok))]
                )
                mean_without_special_tokens = np.mean(ex_scores_without_special_tokens).item()
                std_without_special_tokens = np.std(ex_scores_without_special_tokens).item()
                threshold = mean_without_special_tokens + (std_threshold * std_without_special_tokens)
                ex_scores = np.where(ex_scores_special_tokens_mask, mean_without_special_tokens, ex_scores_squeezed)
            else:
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
            ex_tgt = ex_tgt.to_numpy()
            ex_scores = scaler.fit_transform(ex_scores.reshape(-1, 1)).squeeze()
            precision, recall, _ = precision_recall_curve(ex_tgt, ex_scores)
            auprc = auc(recall, precision)
            f1_val = f1_score(ex_tgt, ex_scores_binary, average="macro")
            mrr_val = mrr(ex_tgt, ex_scores)
            dot_val = dot(ex_tgt, ex_scores)
            auprc_scores.append(auprc)
            f1_scores.append(f1_val)
            mrr_scores.append(mrr_val)
            dot_scores.append(dot_val)
            all_preds += ex_scores_binary.reshape(1, -1).squeeze().tolist()
        auprc_low, auprc_high = conf_bounds(auprc_scores)
        f1_low, f1_high = conf_bounds(f1_scores)
        mrr_low, mrr_high = conf_bounds(mrr_scores)
        dot_low, dot_high = conf_bounds(dot_scores)
        return (
            {
                "avg_auprc": np.mean(auprc_scores).round(4),
                "avg_macro_f1": np.mean(f1_scores).round(4),
                "avg_mrr": np.nanmean(mrr_scores).round(4),
                "avg_dot": np.mean(dot_scores).round(4),
                "auprc_low": auprc_low,
                "auprc_high": auprc_high,
                "std_auprc": np.std(auprc_scores).round(4),
                "f1_low": f1_low,
                "f1_high": f1_high,
                "std_macro_f1": np.std(f1_scores).round(4),
                "mrr_low": mrr_low,
                "mrr_high": mrr_high,
                "std_mrr": np.nanstd(mrr_scores).round(4),
                "dot_low": dot_low,
                "dot_high": dot_high,
                "std_dot": np.std(dot_scores).round(4),
            },
            all_preds,
            {
                "auprc": list(auprc_scores),
                "macro_f1": list(f1_scores),
                "mrr": list(mrr_scores),
                "dot": list(dot_scores),
            },
        )
    else:
        tgt = df[target_column]
        if valid_pos is not None:
            tgt = [s if df[pos_column][i] in valid_pos else 0 for i, s in enumerate(tgt)]
        if initial_only:
            tgt = [s if initial_char in df[token_column][i] else 0 for i, s in enumerate(tgt)]
        if do_random:
            np.random.seed(42)
            scores = np.random.rand(len(tgt)) * 0.5
            matches_pos = np.random.choice(len(tgt), n_random_matches_per_example, replace=False)
            scores[matches_pos] += 0.5
        else:
            scores = df[score_column].to_numpy()
        # Select only scores one standard deviation away from the mean
        scores_binary = scores > (scores.mean() + (std_threshold * scores.std()))
        tgt = tgt.to_numpy()
        scores = scaler.fit_transform(scores.reshape(-1, 1)).squeeze()
        precision, recall, _ = precision_recall_curve(tgt, scores)
        auprc = auc(recall, precision)
        f1_val = f1_score(tgt, scores_binary, average="macro")
        return (
            {
                "auprc": auprc.round(4),
                "macro_f1": f1_val.round(4),
            },
            scores_binary.tolist(),
            None,
        )
