from typing import Dict, List, Optional, Protocol, Tuple

import inseq
import pandas as pd
import torch
from inseq import AttributionModel, FeatureAttributionOutput
from inseq.attr.step_functions import StepFunctionArgs, _get_contrast_output
from inseq.data import FeatureAttributionInput
from inseq.utils import logits_kl_divergence

from .data_utils import DatasetExample
from .enums import AttributeFnEnum
from .model_utils import has_lang_tag


def kl_div_per_layer_fn(
    args: StepFunctionArgs,
    contrast_target_prefixes: Optional[FeatureAttributionInput] = None,
    contrast_sources: Optional[FeatureAttributionInput] = None,
    contrast_targets: Optional[FeatureAttributionInput] = None,
    contrast_targets_alignments: Optional[List[List[Tuple[int, int]]]] = None,
    top_k: int = 0,
    top_p: float = 1.0,
    min_tokens_to_keep: int = 1,
):
    """Compute the KL divergence between original and contrastive probabilities at every layer of the model using the
    logit lens approach to project intermediate hidden states to logits.
    """

    original_batch = args.attribution_model.formatter.convert_args_to_batch(args)
    original_output = args.attribution_model.get_forward_output(
        original_batch,
        output_hidden_states=True,
    )
    contrast_output = _get_contrast_output(
        args,
        contrast_sources=contrast_sources,
        contrast_target_prefixes=contrast_target_prefixes,
        contrast_targets=contrast_targets,
        contrast_targets_alignments=contrast_targets_alignments,
        output_hidden_states=True,
    )
    all_kl_divergences = []
    for i in range(len(original_output.decoder_hidden_states) - 1):
        original_logits = args.attribution_model.model.lm_head(original_output.decoder_hidden_states[i][:, -1, :])
        contrast_logits = args.attribution_model.model.lm_head(contrast_output.decoder_hidden_states[i][:, -1, :])
        original_logits = original_logits + args.attribution_model.model.final_logits_bias
        contrast_logits = contrast_logits + args.attribution_model.model.final_logits_bias
        kl_divergence = logits_kl_divergence(
            original_logits=original_logits,
            contrast_logits=contrast_logits,
            top_p=top_p,
            top_k=top_k,
            min_tokens_to_keep=min_tokens_to_keep,
        )
        all_kl_divergences.append(kl_divergence)
    return torch.stack(all_kl_divergences, dim=1)


# Register the function defined above
# Since outputs are still probabilities, contiguous tokens can still be aggregated using product
inseq.register_step_function(fn=kl_div_per_layer_fn, identifier="kl_div_per_layer", overwrite=True)


class AttributeFn(Protocol):
    def __call__(
        self,
        example: DatasetExample,
        model: AttributionModel,
        curr_idx: int,
        use_gold_target_current: bool,
        use_gold_target_context: bool,
    ) -> pd.DataFrame:
        ...


def out_to_df(out: FeatureAttributionOutput, idx: int) -> pd.DataFrame:
    df = pd.DataFrame(out.get_scores_dicts()[0]["step_scores"])
    df = df.transpose().reset_index().rename(columns={"level_0": "token_idx", "level_1": "token"})
    df.insert(0, "example_idx", idx)
    return df


def base_attribute_fn(
    example: DatasetExample,
    model: AttributionModel,
    curr_idx: int,
    use_gold_target_current: bool,
    use_gold_target_context: bool,
) -> pd.DataFrame:
    target_current = example.gold_target_current if use_gold_target_current else example.generated_target_current
    target_context = example.gold_target_context if use_gold_target_context else example.generated_target_context
    has_target_context = target_context is not None and pd.notnull(target_context)
    out = model.attribute(
        example.source_current,
        target_current,
        attribute_target=True,
        step_scores=["probability", "contrast_prob", "contrast_prob_diff", "pcxmi", "kl_divergence"],
        contrast_sources=example.source_full,
        contrast_target_prefixes=target_context if has_target_context else None,
        show_progress=False,
    )
    return out_to_df(out, curr_idx)


def top_p_attribute_fn(
    example: DatasetExample,
    model: AttributionModel,
    curr_idx: int,
    use_gold_target_current: bool,
    use_gold_target_context: bool,
) -> pd.DataFrame:
    overall_df = None
    target_current = example.gold_target_current if use_gold_target_current else example.generated_target_current
    target_context = example.gold_target_context if use_gold_target_context else example.generated_target_context
    has_target_context = target_context is not None and pd.notnull(target_context)
    for top_p in [0.1, 0.3, 0.5, 0.7, 0.9]:
        out = model.attribute(
            example.source_current,
            target_current,
            attribute_target=True,
            step_scores=["kl_divergence", "top_p_size"],
            contrast_sources=example.source_full,
            contrast_target_prefixes=target_context if has_target_context else None,
            show_progress=False,
            top_p=top_p,
        )
        df = out_to_df(out, curr_idx)
        if overall_df is None:
            overall_df = df.rename(
                columns={"kl_divergence": f"kl_div_{int(top_p * 100)}", "top_p_size": f"top_p_size_{int(top_p * 100)}"}
            )
        else:
            overall_df[f"kl_div_{int(top_p * 100)}"] = df["kl_divergence"]
            overall_df[f"top_p_size_{int(top_p * 100)}"] = df["top_p_size"]
    return overall_df


def logit_lens_attribute_fn(
    example: DatasetExample,
    model: AttributionModel,
    curr_idx: int,
    use_gold_target_current: bool,
    use_gold_target_context: bool,
) -> pd.DataFrame:
    target_current = example.gold_target_current if use_gold_target_current else example.generated_target_current
    target_context = example.gold_target_context if use_gold_target_context else example.generated_target_context
    has_target_context = target_context is not None and pd.notnull(target_context)
    out = model.attribute(
        example.source_current,
        target_current,
        attribute_target=True,
        step_scores=["kl_div_per_layer"],
        contrast_sources=example.source_full,
        contrast_target_prefixes=target_context if has_target_context else None,
        show_progress=False,
    )
    for layer_idx in range(out[0].step_scores["kl_div_per_layer"].shape[0]):
        out[0].step_scores[f"kl_div_l{layer_idx}"] = out[0].step_scores["kl_div_per_layer"][layer_idx, :]
    del out[0].step_scores["kl_div_per_layer"]
    return out_to_df(out, curr_idx)


def attribute_contrast(
    example: DatasetExample,
    model: AttributionModel,
    curr_idx: int,
    use_gold_target_current: bool,
    use_gold_target_context: bool,
    context_separator: str = "<brk>",
    attributed_fn: str = "contrast_prob_diff",
    attribution_method: str = "saliency",
) -> Optional[FeatureAttributionOutput]:
    target_current = example.gold_target_current if use_gold_target_current else example.generated_target_current
    target_context = example.gold_target_context if use_gold_target_context else example.generated_target_context
    has_target_context = target_context is not None and pd.notnull(target_context)
    # Handle missing source context
    if example.source_full.startswith(context_separator) or not isinstance(target_current, str):
        print(f"Skipping example {curr_idx}")
        return None
    target_full = target_context + f"{context_separator} " + target_current if has_target_context else target_current
    offset = 0
    if has_target_context:
        target_context_tokens = model.encode(target_context, as_targets=True).input_tokens[0]
        offset = len(target_context_tokens)
        if has_lang_tag(model):
            offset -= 1
    curr_len = len(model.encode(target_current, as_targets=True).input_tokens[0]) - 1
    out = model.attribute(
        example.source_full,
        target_full,
        attribute_target=True,
        show_progress=False,
        attr_pos_start=offset if has_target_context else None,
        attributed_fn=attributed_fn,
        method=attribution_method,
        contrast_sources=example.source_current,
        contrast_targets=target_current,
        contrast_targets_alignments=[
            (idx_full, idx_curr)
            for idx_curr, idx_full in enumerate(range(offset, offset + curr_len), start=1 if has_target_context else 0)
        ],
    )
    return out


def input_contributions_attribute_fn(
    example: DatasetExample,
    model: AttributionModel,
    curr_idx: int,
    use_gold_target_current: bool,
    use_gold_target_context: bool,
    context_separator: str = "<brk>",
    attributed_fn: str = "contrast_prob_diff",
    attribution_method: str = "saliency",
) -> pd.DataFrame:
    target_context = example.gold_target_context if use_gold_target_context else example.generated_target_context
    has_target_context = target_context is not None and pd.notnull(target_context)
    out = attribute_contrast(
        example=example,
        model=model,
        curr_idx=curr_idx,
        use_gold_target_current=use_gold_target_current,
        use_gold_target_context=use_gold_target_context,
        context_separator=context_separator,
        attributed_fn=attributed_fn,
        attribution_method=attribution_method,
    )
    if out is None:
        return None
    # Aggregate contributions for context and current source tokens, keeping special tokens separate
    # If target context is used, target context is also aggregated.
    aggr_args = {}
    use_lang_tag = has_lang_tag(model)
    source_sep_idx = [t.token for t in out[0].source].index(context_separator)
    lang_tag_offset = 1 if use_lang_tag else 0
    aggr_args["source_spans"] = [(lang_tag_offset, source_sep_idx), (source_sep_idx + 1, len(out[0].source) - 1)]
    if has_target_context:
        special_tok = context_separator
        if use_lang_tag:
            special_tok = f"{model.tokenizer.tgt_lang} → {context_separator}"
        target_sep_idx = [t.token for t in out[0].target].index(special_tok)
        aggr_args["target_spans"] = [(lang_tag_offset, target_sep_idx)]
    aggr_out = out.aggregate("spans", **aggr_args).aggregate()
    assert aggr_out[0].source_attributions.size(0) == 4 + lang_tag_offset, (
        f"Expected {4 + lang_tag_offset} source tokens but found {aggr_out[0].source_attributions.size(0)} "
        f"instead: {aggr_out[0].source}"
    )
    if use_lang_tag:
        aggr_out[0].step_scores["src_langtag_attr"] = aggr_out[0].source_attributions[0, :]
    aggr_out[0].step_scores["src_ctx_attr"] = aggr_out[0].source_attributions[0 + lang_tag_offset, :]
    aggr_out[0].step_scores["src_brk_attr"] = aggr_out[0].source_attributions[1 + lang_tag_offset, :]
    aggr_out[0].step_scores["src_curr_attr"] = aggr_out[0].source_attributions[2 + lang_tag_offset, :]
    aggr_out[0].step_scores["src_eos_attr"] = aggr_out[0].source_attributions[3 + lang_tag_offset, :]
    tgt_curr_start_idx = 0
    if use_lang_tag:
        aggr_out[0].step_scores["tgt_langtag_attr"] = aggr_out[0].target_attributions[tgt_curr_start_idx, :]
        tgt_curr_start_idx += 1
    if has_target_context:
        aggr_out[0].step_scores["tgt_ctx_attr"] = aggr_out[0].target_attributions[tgt_curr_start_idx, :]
        aggr_out[0].step_scores["tgt_brk_attr"] = aggr_out[0].target_attributions[tgt_curr_start_idx + 1, :]
        tgt_curr_start_idx += 2
    aggr_out[0].step_scores["tgt_curr_attr"] = aggr_out[0].target_attributions[tgt_curr_start_idx:, :].nansum(axis=0)
    assert torch.allclose(
        torch.stack(list(aggr_out[0].step_scores.values()), dim=1).nansum(axis=1),
        torch.ones_like(aggr_out[0].step_scores["src_ctx_attr"]),
    )
    df = pd.DataFrame(aggr_out.get_scores_dicts(do_aggregation=False)[0]["step_scores"])
    df = df.transpose().reset_index().rename(columns={"level_0": "token_idx", "level_1": "token"})
    if has_target_context:
        df["token_idx"] = list(range(len(aggr_out[0].step_scores["src_ctx_attr"])))
    df.insert(0, "example_idx", curr_idx)
    return df


ATTRIBUTE_FN_DICT: Dict[str, AttributeFn] = {
    attribute_fn_name: globals()[f"{attribute_fn_name}_attribute_fn"] for attribute_fn_name in AttributeFnEnum
}


def get_attribute_fn(attribute_fn: str) -> AttributeFn:
    if attribute_fn not in ATTRIBUTE_FN_DICT:
        raise ValueError(f"Unknown attribute function {attribute_fn}")
    return ATTRIBUTE_FN_DICT[attribute_fn]
