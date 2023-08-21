import argparse
import logging
from pathlib import Path

import inseq
import pandas as pd
from pecore.alignment_utils import get_match_from_contrastive_pair, get_model_cue_target_tags
from pecore.data_utils import DatasetExample
from pecore.enums import ModelTypeEnum
from pecore.inseq_utils import get_imputation_scores_df
from pecore.model_utils import get_lang_from_model_type, has_lang_tag
from tqdm import tqdm

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
        default="outputs/scores",
        help="Root folder to save processed examples outputs",
    )
    parser.add_argument(
        "--examples_path",
        type=str,
        required=True,
        help="Path to the file containing examples to process.",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Index to start processing examples from.",
    )
    parser.add_argument(
        "--max_idx",
        type=int,
        default=None,
        help="Maximum index to process examples to.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HF Hub model identifier to load the model for computing scores.",
    )
    parser.add_argument(
        "--attribution_methods",
        type=str,
        nargs="+",
        choices=inseq.list_feature_attribution_methods(),
        default=["saliency", "attention", "input_x_gradient"],
        help="Attribution method to use for computing scores.",
    )
    parser.add_argument(
        "--attributed_fns",
        type=str,
        nargs="+",
        choices=inseq.list_step_functions(),
        default=["contrast_prob_diff", "kl_divergence"],
        help="Target attributed function to use for computing scores.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=[t.value for t in ModelTypeEnum],
        default=None,
        help="Model type to use, required for models using language tags.",
    )
    parser.add_argument(
        "--src_lang",
        type=str,
        default="eng",
        help="Source language, required for models using language tags.",
    )
    parser.add_argument(
        "--tgt_lang",
        type=str,
        default="fra",
        help="Target language, required for models using language tags.",
    )
    parser.add_argument(
        "--use_gold_target_current",
        action="store_true",
        help="Use gold target current sentence instead of model generation.",
    )
    parser.add_argument(
        "--use_gold_target_context",
        action="store_true",
        help="Use gold target context instead of model generation.",
    )
    parser.add_argument(
        "--skip_token_tags",
        action="store_true",
        help="Set to skip adding extra columns to mark tagged tokens.",
    )
    parser.add_argument(
        "--skip_example_status",
        action="store_true",
        help="Set to skip adding an extra column to store the classification status of the selected example",
    )
    args = parser.parse_args()
    return args


def tag_cci_metrics():
    args = parse_args()
    model = inseq.load_model(args.model_name, "dummy")
    examples = pd.read_csv(args.examples_path, sep="\t").to_dict("records")
    examples = [DatasetExample(**ex) for ex in examples]
    dataset_name = Path(args.examples_path).stem.split("-")[0]
    model_id = Path(args.examples_path).stem.split(".")[0][len(dataset_name) + 1 :]
    out_path = Path(args.output_dir) / f"{dataset_name}-{model_id}-cci.tsv"
    scores_df = None
    if args.start_idx > 0:
        scores_df = pd.read_csv(out_path, sep="\t")
    if args.max_idx is None:
        args.max_idx = len(examples)
    if has_lang_tag(model):
        model.tokenizer.src_lang = get_lang_from_model_type(args.model_type, args.src_lang)
        model.tokenizer.tgt_lang = get_lang_from_model_type(args.model_type, args.tgt_lang)
    for idx, ex in tqdm(enumerate(examples), total=args.max_idx):
        if idx < args.start_idx:
            continue
        if idx >= args.max_idx:
            break
        curr_df = get_imputation_scores_df(
            example=ex,
            model=model,
            curr_idx=idx,
            use_gold_target_current=args.use_gold_target_current,
            use_gold_target_context=args.use_gold_target_context,
            model_type=args.model_type,
            attribution_methods=args.attribution_methods,
            attributed_fns=args.attributed_fns,
            # TODO: Provide custom target tags to test without gold tags.
        )
        if curr_df is None:
            logger.warning("Skipping example %d, no imputation scores available.", idx)
            continue
        if not args.skip_token_tags:
            target_context = ex.gold_target_context if args.use_gold_target_context else ex.generated_target_context
            has_target_context = target_context is not None and pd.notnull(target_context)
            if ex.source_context_tagged is None or (ex.gold_target_context_tagged is None and has_target_context):
                raise ValueError(
                    f"Example {idx} does not have tagged gold source and target contexts, cannot add cue tags."
                    "Please provide a tagged version of the example or set --skip_token_tags."
                )
            try:
                cue_tags, _ = get_model_cue_target_tags(
                    ex.source_context_tagged,
                    ex.source_context,
                    model,
                    is_generated_untagged=False,
                    model_type=args.model_type,
                    is_target=False,
                    is_current=False,
                    add_lang_tag=False,
                )
                if has_target_context:
                    target_cue_tags, _ = get_model_cue_target_tags(
                        ex.gold_target_context_tagged,
                        target_context,
                        model,
                        is_generated_untagged=not args.use_gold_target_context,
                        model_type=args.model_type,
                        is_current=False,
                        add_lang_tag=False,
                    )
                    cue_tags = cue_tags + target_cue_tags
                curr_df["is_supporting_context"] = cue_tags
            except Exception as e:
                logger.error(f"Error tagging cue tokens for example {idx}: {e}")
                continue
        # Add an is_example_correct column to the scores file, marking examples where the model correctly disambiguates the
        # gender in the target sentence. To analyze data folds and not as feature, since it is not available in test.
        if not args.skip_example_status:
            if (
                ex.gold_target_current is None
                or ex.generated_target_current is None
                or ex.gold_target_current_contrast is None
            ):
                raise ValueError(
                    "A generated target and a minimal pair of original and contrastive gold targets are required to"
                    " compute the example status.\n Provide examples containing these or set --skip_example_status."
                )
            correct_matches = get_match_from_contrastive_pair(
                ref_text=ex.gold_target_current,
                contrast_ref_text=ex.gold_target_current_contrast,
                pred_text=ex.generated_target_current,
            )
            is_correct = 1 if sum(correct_matches) > 0 else 0
            curr_df["is_example_correct"] = is_correct
        if scores_df is None:
            scores_df = curr_df
        else:
            scores_df = pd.concat([scores_df, curr_df], ignore_index=True)
        scores_df.round(4).to_csv(out_path, index=False, sep="\t")


if __name__ == "__main__":
    tag_cci_metrics()
