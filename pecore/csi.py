from typing import Callable, Dict, Optional, Tuple

import inseq
import pandas as pd

from .alignment_utils import get_model_cue_target_tags
from .data_utils import get_formatted_examples
from .inseq_utils import base_attribute_fn


def get_model_id_and_name(
    cwd: int = 1, ctx: int = 4, model_type: str = "", dataset: str = "scat", model_name: str = None
) -> Tuple[str, str]:
    if model_name is None:
        if not model_type:
            raise ValueError("Must specify model_type or model_name")
        model_id = f"{dataset}-{model_type}-ctx{ctx}-cwd{cwd}"
        model_name = f"context-mt/{model_id}-en-fr"
    else:
        model_id = model_type
    return model_id, model_name


def context_sensitive_span_identification_scores(
    examples_path: Optional[str] = None,
    cwd: int = 1,
    ctx: int = 4,
    model_type: str = "",
    dataset: str = "scat",
    model_name: Optional[str] = None,
    force_gen: bool = False,
    has_context: bool = True,
    has_lang_tag: bool = False,
    has_target_context: bool = False,
    start_idx: int = 0,
    max_idx: Optional[int] = None,
    add_tags: bool = True,
    attribute_fn: Callable[[Dict[str, str], int], pd.DataFrame] = base_attribute_fn,
) -> pd.DataFrame:
    if examples_path is None:
        model_id, model_name = get_model_id_and_name(
            cwd=cwd, ctx=ctx, model_type=model_type, dataset=dataset, model_name=model_name
        )
        examples = get_formatted_examples(
            model_name=model_name,
            force_gen=force_gen,
            has_context=has_context,
            has_lang_tag=has_lang_tag,
            has_target_context=has_target_context,
            start_idx=start_idx,
            max_idx=max_idx,
        )
    else:
        examples = pd.read_csv(examples_path, sep="\t").to_dict("records")
        model_id = model_type
    scores_df = None
    if start_idx > 0:
        scores_df = pd.read_csv(
            f"translations/scores/temp/{model_id}-scores-{'gold' if force_gen else 'gen'}.tsv", sep="\t"
        )
    if max_idx is None:
        max_idx = len(examples)
    model = inseq.load_model(model_name, "saliency")
    if has_lang_tag:
        model.tokenizer.src_lang = "en_XX"
        model.tokenizer.tgt_lang = "fr_XX"
    for idx, ex in enumerate(examples):
        if idx < start_idx:
            continue
        if idx >= max_idx:
            break
        df = attribute_fn(ex, model, idx)
        if df is None:
            continue
        if add_tags:
            try:
                if force_gen:
                    cue_tags, target_tags = get_model_cue_target_tags(ex["orig_fr_with_tags"], ex["orig_fr"], model)
                else:
                    cue_tags, target_tags = get_model_cue_target_tags(ex["fr_with_tags"], ex["fr"], model)
                df["is_supporting_context"] = cue_tags
                df["is_context_sensitive"] = target_tags
            except Exception as ex:
                print(f"Excluding example {idx} due to error {ex}")
                continue
        if scores_df is None:
            scores_df = df
        else:
            scores_df = pd.concat([scores_df, df], axis=0)
        scores_df.to_csv(
            f"translations/scores/temp/{model_id}-scores-{'gold' if force_gen else 'gen'}.tsv", index=False, sep="\t"
        )
