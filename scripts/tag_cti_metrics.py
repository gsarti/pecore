import argparse
import logging
from inspect import getfullargspec
from pathlib import Path

import inseq
import pandas as pd
from pecore.alignment_utils import get_match_from_contrastive_pair, get_model_cue_target_tags
from pecore.data_utils import DatasetExample
from pecore.enums import AttributeFnEnum, ModelTypeEnum
from pecore.inseq_utils import get_attribute_fn
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
        "--attribute_fns",
        type=str,
        nargs="+",
        choices=[t.value for t in AttributeFnEnum],
        default=[
            AttributeFnEnum.BASE,
            AttributeFnEnum.TOP_P,
            AttributeFnEnum.LOGIT_LENS,
            AttributeFnEnum.INPUT_CONTRIBUTIONS,
        ],
        help="Attribute functions to use to compute scores.",
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
        "--skip_token_tags",
        action="store_true",
        help="Set to skip adding extra columns to mark tagged tokens.",
    )
    parser.add_argument(
        "--skip_example_status",
        action="store_true",
        help="Set to skip adding an extra column to store the classification status of the selected example",
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
        "--context_separator",
        type=str,
        default="<brk>",
        help="Source language, required for models using language tags.",
    )
    args = parser.parse_args()
    return args


def check_examples(args: argparse.Namespace, ex: DatasetExample):
    if args.use_gold_target_current and ex.gold_target_current is None:
        raise ValueError(
            "Cannot use gold target current sentence, gold_target_current is not available in the examples.\n"
            "Please provide examples with gold targets or remove --use_gold_target_current."
        )
    if not args.use_gold_target_current and ex.generated_target_current is None:
        raise ValueError(
            "Cannot use model generated target current sentence, generated_target_current is not available in the"
            " examples.\nPlease provide examples with generated targets or set --use_gold_target_current."
        )
    if args.use_gold_target_context and ex.gold_target_context is None:
        raise ValueError(
            "Cannot use gold target context, gold_target_context is not available in the examples.\n"
            "Please provide examples with gold targets or remove --use_gold_target_context.\n"
            "NOTE: Only models generating context alongside current targets can use target context."
        )
    if not args.use_gold_target_context and ex.generated_target_context is None:
        raise ValueError(
            "Cannot use model generated target context, generated_target_context is not available in the examples.\n"
            "Please provide examples with generated targets or set --use_gold_target_context.\n"
            "NOTE: Only models generating context alongside current targets can use target context."
        )


def tag_cti_metrics():
    args = parse_args()
    model = inseq.load_model(args.model_name, "dummy")
    examples = pd.read_csv(args.examples_path, sep="\t").to_dict("records")
    examples = [DatasetExample(**ex) for ex in examples]
    check_examples(args, examples[0])
    dataset_name = Path(args.examples_path).stem.split("-")[0]
    model_id = Path(args.examples_path).stem.split(".")[0][len(dataset_name) + 1 :]
    scores_df = None
    out_path = Path(args.output_dir) / f"{dataset_name}-{model_id}-cti.tsv"
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
        curr_full_scores_df = None
        for attribute_fn_name in args.attribute_fns:
            attribute_fn_args = {}
            if not pd.notnull(ex.gold_target_current if args.use_gold_target_current else ex.generated_target_current):
                continue
            attribute_fn = get_attribute_fn(attribute_fn_name)
            params = getfullargspec(attribute_fn)
            if "context_separator" in params.args:
                attribute_fn_args["context_separator"] = args.context_separator
            curr_partial_scores_df = attribute_fn(
                example=ex,
                model=model,
                curr_idx=idx,
                use_gold_target_current=args.use_gold_target_current,
                use_gold_target_context=args.use_gold_target_context,
                **attribute_fn_args,
            )
            if curr_partial_scores_df is None:
                continue
            if curr_full_scores_df is None:
                curr_full_scores_df = curr_partial_scores_df
            else:
                curr_full_scores_df = pd.merge(
                    curr_full_scores_df, curr_partial_scores_df, on=["example_idx", "token_idx", "token"], how="left"
                )
        if curr_full_scores_df is None:
            continue
        curr_full_scores_df.fillna(0)

        # Add columns to mark tagged tokens.
        if not args.skip_token_tags:
            if ex.gold_target_current_tagged is None:
                raise ValueError(
                    f"Example {idx} does not have gold_target_current_tagged, cannot add token tags."
                    "Please provide a tagged version of the example or set --skip_token_tags."
                )
            try:
                cue_tags, target_tags = get_model_cue_target_tags(
                    ex.gold_target_current_tagged,
                    ex.gold_target_current if args.use_gold_target_current else ex.generated_target_current,
                    model,
                    is_generated_untagged=not args.use_gold_target_current,
                    model_type=args.model_type,
                )
                curr_full_scores_df["is_supporting_context"] = cue_tags
                curr_full_scores_df["is_context_sensitive"] = target_tags
            except Exception as e:
                print(f"Excluding example {idx} due to error {e}")
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
            curr_full_scores_df["is_example_correct"] = is_correct
        if scores_df is None:
            scores_df = curr_full_scores_df
        else:
            scores_df = pd.concat([scores_df, curr_full_scores_df], ignore_index=True)
        scores_df.round(4).to_csv(out_path, index=False, sep="\t")


if __name__ == "__main__":
    tag_cti_metrics()
