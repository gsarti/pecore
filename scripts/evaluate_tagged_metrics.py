import argparse
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd
from pecore.analysis_utils import (
    get_cti_mix_features,
    get_metric_results_from_scores,
    get_metrics_result_with_trained_model,
    get_splits,
)
from pecore.enums import CCIMetricsEnum, CTIMetricsEnum, EvalModeEnum, TaggedDatasetEnum

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s  %(message)s",
    datefmt="%d-%m-%y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/metrics_evals",
        help="Root folder to save scores evaluations.",
    )
    parser.add_argument(
        "--scores_path",
        type=str,
        required=True,
        help="Path to the file containing scores to evaluate.",
    )
    parser.add_argument(
        "--example_correct_column",
        type=str,
        default="is_example_correct",
        help="Column name identifying the column marking the correct example.",
    )
    parser.add_argument(
        "--example_target_column",
        type=str,
        default="is_context_sensitive",
        help="Column name identifying the column marking the classification target.",
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        required=True,
        choices=[e.value for e in EvalModeEnum],
        help="Evaluation mode to use.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[e.value for e in TaggedDatasetEnum],
        help="Dataset to use for evaluation.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=[],
        help="Metrics to evaluate.",
    )
    parser.add_argument(
        "--model_n_layers",
        type=int,
        default=6,
        help="Number of layers of the model.",
    )
    parser.add_argument(
        "--model_n_heads",
        type=int,
        default=8,
        help="Number of layers of the model.",
    )
    parser.add_argument(
        "--has_target_context",
        action="store_true",
        help="Whether the model has target context.",
    )
    parser.add_argument(
        "--use_trained_model",
        action="store_true",
        help="Whether to use a trained model or raw scores for evaluation.",
    )
    parser.add_argument(
        "--initial_only",
        action="store_true",
        help="Whether to consider only the initial subword of a word as valid target.",
    )
    parser.add_argument(
        "--valid_input_types",
        type=str,
        nargs="+",
        default=["S", "T"],
        help="Valid input sides for contribution to consider. Default all.",
    )
    parser.add_argument(
        "--valid_pos_tags",
        type=str,
        nargs="+",
        default=None,
        help="Valid POS tags for contribution to consider. Default all.",
    )
    parser.add_argument(
        "--average_example_scores",
        action="store_true",
        help=(
            "Whether to average the scores over examples. If not set, scores are computed over all token at once"
            " instead."
        ),
    )
    parser.add_argument(
        "--special_tokens_to_remove",
        type=str,
        nargs="+",
        default=["fr_XX"],
        help="Special tokens to remove from the input for measuring scores.",
    )
    parser.add_argument(
        "--save_preds",
        action="store_true",
        help="Whether to save predictions to a file.",
    )
    args = parser.parse_args()
    args.processed_metrics = defaultdict(list)
    if not args.metrics:
        if args.eval_mode == EvalModeEnum.CTI:
            args.metrics = [e.value for e in CTIMetricsEnum]
        elif args.eval_mode == EvalModeEnum.CCI:
            args.metrics = [e.value for e in CCIMetricsEnum]
    for metric in args.metrics:
        if (args.eval_mode == EvalModeEnum.CTI.value and metric not in [e.value for e in CTIMetricsEnum]) or (
            args.eval_mode == EvalModeEnum.CCI.value and metric not in [e.value for e in CCIMetricsEnum]
        ):
            raise ValueError(f"Metric {metric} is not valid for evaluation mode {args.eval_mode}.")
        if metric == CTIMetricsEnum.CTI_MIX:
            cti_mix = get_cti_mix_features(args.model_n_layers, args.has_target_context)
            args.processed_metrics[CTIMetricsEnum.CTI_MIX].append(cti_mix)
        elif metric == CCIMetricsEnum.ATTN_BEST:
            for l in range(args.model_n_layers):
                for h in range(args.model_n_heads):
                    args.processed_metrics[CCIMetricsEnum.ATTN_BEST].append([f"attention_{l}l_{h}h"])
        else:
            args.processed_metrics[metric].append([metric])
    return args


def evaluate_tagged_metrics():
    args = parse_args()
    scores_df = pd.read_csv(args.scores_path, sep="\t")
    dataset_name = Path(args.scores_path).stem.split("-")[0]
    model_id = Path(args.scores_path).stem.split(".")[0][len(dataset_name) + 1 :]
    input_type_name = "" if args.valid_input_types == ["S", "T"] else "-" + "-".join(args.valid_input_types)
    pos_name = "" if args.valid_pos_tags is None else "-" + "-".join(args.valid_pos_tags)
    avg_example_name = "-avg" if args.average_example_scores else ""
    initial_only_name = "-initial" if args.initial_only else ""
    model_name = "-model" if args.use_trained_model else ""
    root_fname = (
        f"{dataset_name}-{model_id}{input_type_name}{pos_name}{avg_example_name}{initial_only_name}{model_name}"
    )
    eval_fname = f"{root_fname}-eval.tsv"
    out_path = Path(args.output_dir) / eval_fname
    splits = get_splits(args.dataset, scores_df, target_column=args.example_target_column, eval_mode=args.eval_mode)
    all_scores = []
    for metric_name, metrics in args.processed_metrics.items():
        for metric in metrics:
            for split_name, split in splits.items():
                logger.info(f"Evaluating {metric_name} ({metric}) on {split_name} split.")
                score_param_name = "scores_columns" if args.use_trained_model else "score_column"
                if args.eval_mode == EvalModeEnum.CTI:
                    if metric_name == CTIMetricsEnum.RANDOM:
                        kwargs = {score_param_name: ["probability"], "do_random": True}
                    else:
                        kwargs = {score_param_name: metric}
                elif args.eval_mode == EvalModeEnum.CCI:
                    if metric_name == CCIMetricsEnum.RANDOM:
                        kwargs = {score_param_name: ["attention_default"], "do_random": True}
                    else:
                        kwargs = {score_param_name: metric}
                if args.use_trained_model:
                    kwargs[score_param_name] = kwargs[score_param_name][0]
                    scores = get_metrics_result_with_trained_model(
                        scores_df,
                        split["train"],
                        split["test"],
                        target_column=args.example_target_column,
                        **kwargs,
                    )
                    preds = None
                else:
                    scores, preds = get_metric_results_from_scores(
                        scores_df,
                        split["test"],
                        target_column=args.example_target_column,
                        initial_only=args.initial_only,
                        average_example_scores=args.average_example_scores,
                        valid_input_types=args.valid_input_types,
                        valid_pos=args.valid_pos_tags,
                        special_tokens_to_remove=args.special_tokens_to_remove,
                        **kwargs,
                    )
                if preds is not None and args.save_preds and split_name == f"{args.dataset}_all":
                    preds_path = Path(args.output_dir) / f"{root_fname}-{metric_name}-preds.txt"
                    split_scores = scores_df[split["test"]]
                    split_scores["preds"] = preds
                    score_lines = []
                    ex_grouped_split_scores = split_scores.groupby("example_idx")
                    for ex_id in ex_grouped_split_scores.groups.keys():
                        ex_df = ex_grouped_split_scores.get_group(ex_id)
                        score_lines.append(" ".join(["1" if p else "0" for p in ex_df["preds"]]))
                    with open(preds_path, "w") as f:
                        f.write("\n".join(score_lines))

                df_scores = pd.DataFrame([scores])
                if 0 < len(metrics) < 2:
                    df_scores.insert(0, "metric", metric_name)
                else:
                    df_scores.insert(0, "metric", metric[0])
                df_scores.insert(0, "split", split_name)
                all_scores.append(df_scores)
    all_scores = pd.concat(all_scores)
    all_scores.to_csv(out_path, sep="\t", index=False)


if __name__ == "__main__":
    evaluate_tagged_metrics()
