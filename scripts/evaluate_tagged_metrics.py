import argparse
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd
from pecore.analysis_utils import get_cti_mix_features, get_metrics_result, get_scat_splits
from pecore.enums import CCIMetricsEnum, CTIMetricsEnum, EvalModeEnum

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
    out_path = Path(args.output_dir) / f"{dataset_name}-{model_id}-eval.tsv"
    scat_splits = get_scat_splits(scores_df, target_column=args.example_target_column, eval_mode=args.eval_mode)
    all_scores = []
    for metric_name, metrics in args.processed_metrics.items():
        for metric in metrics:
            if metric_name in (CTIMetricsEnum.RANDOM, CCIMetricsEnum.RANDOM):
                if args.eval_mode == EvalModeEnum.CTI:
                    kwargs = {"scores_columns": ["probability"], "do_random": True}
                elif args.eval_mode == EvalModeEnum.CCI:
                    kwargs = {"scores_columns": ["attention_base"], "do_random": True}
            else:
                kwargs = {"scores_columns": metric}
            for split_name, split in scat_splits.items():
                logger.info(f"Evaluating {metric_name} ({metric}) on {split_name} split.")
                scores = get_metrics_result(
                    scores_df,
                    split["train"],
                    split["test"],
                    target_column=args.example_target_column,
                    **kwargs,
                )
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
