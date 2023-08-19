import argparse
import logging
from pathlib import Path

import inseq
import pandas as pd
from pecore.alignment_utils import get_aligned_gender_annotations, get_model_cue_target_tags
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
    args = parser.parse_args()
    return args


def tag_csi_metrics():
    args = parse_args()
    model = inseq.load_model(args.model_name, "dummy")
    examples = pd.read_csv(args.examples_path, sep="\t").to_dict("records")
    examples = [DatasetExample(**ex) for ex in examples]
    dataset_name = Path(args.examples_path).stem.split("-")[0]
    model_id = Path(args.examples_path).stem.split(".")[0][len(dataset_name) + 1 :]
    scores_df = None
    out_path = Path(args.output_dir) / f"{dataset_name}-{model_id}-csi.tsv"
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
            attribute_fn = get_attribute_fn(attribute_fn_name)
            curr_partial_scores_df = attribute_fn(
                example=ex,
                model=model,
                curr_idx=idx,
                use_gold_target_current=args.use_gold_target_current,
                use_gold_target_context=args.use_gold_target_context,
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
            except Exception as ex:
                print(f"Excluding example {idx} due to error {ex}")
                continue

        # Add an is_example_correct column to the scores file, marking examples where the model correctly disambiguates the
        # gender in the target sentence. To analyze data folds and not as feature, since it is not available in test.
        if not args.skip_example_status:
            correct_matches = get_aligned_gender_annotations(
                ref_text=ex.gold_target_current,
                contrast_ref_text=ex.gold_target_current_contrast,
                mt_text=ex.generated_target_current,
            )
            is_correct = 1 if sum(correct_matches) > 0 else 0
            curr_full_scores_df["is_example_correct"] = is_correct
        if scores_df is None:
            scores_df = curr_full_scores_df
        else:
            scores_df = pd.concat([scores_df, curr_full_scores_df], ignore_index=True)
        scores_df.to_csv(out_path, index=False, sep="\t")


if __name__ == "__main__":
    tag_csi_metrics()