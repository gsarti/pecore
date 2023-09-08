import logging
import warnings
from itertools import product
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

import inseq
import pandas as pd
import torch
from inseq import AttributionModel, FeatureAttributionOutput
from inseq.attr.step_functions import StepFunctionArgs, _get_contrast_output
from inseq.data import FeatureAttributionInput
from inseq.utils import logits_kl_divergence

from .alignment_utils import get_model_cue_target_tags, tokenize_subwords
from .data_utils import DatasetExample
from .enums import AttributeFnEnum
from .model_utils import has_lang_tag

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

inseq_aggr_logger = logging.getLogger("inseq.data.aggregator")
inseq_aggr_logger.setLevel(logging.WARNING)
inseq_align_logger = logging.getLogger("inseq.utils.alignment_utils")
inseq_align_logger.setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


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
        **kwargs,
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
    model_has_lang_tag = has_lang_tag(model)
    # Handle missing source context
    if (context_separator and example.source_full.startswith(context_separator)) or not isinstance(
        target_current, str
    ):
        print(f"Skipping example {curr_idx}")
        return None
    target_full = target_context + f"{context_separator} " + target_current if has_target_context else target_current
    offset = 0
    if has_target_context:
        target_context_tokens = model.encode(
            target_context, as_targets=True, add_bos_token=not model_has_lang_tag
        ).input_tokens[0]
        offset = len(target_context_tokens) - 1
        if context_separator:
            offset += 1
    curr_len = len(model.encode(target_current, as_targets=True, add_bos_token=not model_has_lang_tag).input_tokens[0])
    start = 1 if has_target_context else 0
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
            (idx_full, idx_curr) for idx_curr, idx_full in enumerate(range(offset, offset + curr_len), start=start)
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


def prepare_cci_params(
    model: AttributionModel,
    has_output_context: bool,
    output_context: str,
    output_current: str,
    input_current: str,
    impute_with_contextless_output: bool,
    ctx_break: str,
    cti_tok_idx: int,
    model_use_ctx_break: bool,
    model_has_lang_tag: bool,
    force_context_aware_output_prefix: bool,
    gen_kwargs: Dict[str, Any] = {},
) -> Tuple[str, Union[List[Tuple[int, int]], str], int, int]:
    lang_tag_offset = 1 if model_has_lang_tag else 0
    if force_context_aware_output_prefix:
        output_current_enc = model.encode(output_current, as_targets=True)
        gen_kwargs["decoder_input_ids"] = output_current_enc.input_ids[:, : cti_tok_idx + 1 + lang_tag_offset]
    offset = 0
    if has_output_context:
        output_context_tokens = model.encode(
            output_context.strip(ctx_break), as_targets=True, add_bos_token=not model_has_lang_tag
        ).input_tokens[0]
        # Drop EOS token
        offset = len(output_context_tokens) - 1
        # Assuming context break is a single token in model's vocabulary
        if model_use_ctx_break:
            offset += 1
    if impute_with_contextless_output:
        output_current_contrast = model.generate(input_current, max_new_tokens=200, **gen_kwargs)[0]
        aligns = "auto"
    else:
        output_current_contrast = output_current
        curr_len = len(
            model.encode(output_current, as_targets=True, add_bos_token=not model_has_lang_tag).input_tokens[0]
        )
        start = 0
        if has_output_context and model_use_ctx_break:
            if model_has_lang_tag:
                start = 2
            else:
                start = 1
        aligns = [
            (idx_full, idx_curr) for idx_curr, idx_full in enumerate(range(offset, offset + curr_len), start=start)
        ]
    # +1 for lang tag if present, BOS otherwise
    pos_start = offset + cti_tok_idx + 1
    return output_current_contrast, aligns, pos_start


def get_imputation_scores_df(
    example: DatasetExample,
    model: AttributionModel,
    curr_idx: int,
    use_gold_target_current: bool,
    use_gold_target_context: bool,
    model_type: str,
    attribution_methods: List[str],
    attributed_fns: List[str],
    context_separator: str = "<brk>",
    target_tags: Optional[List[bool]] = None,
    include_per_unit_scores: bool = False,
    units_names: List[str] = ["l", "h"],
    impute_with_contextless_output: bool = False,
    force_context_aware_output_prefix: bool = False,
    gen_kwargs: Dict[str, Any] = {},
) -> Optional[pd.DataFrame]:
    target_context = example.gold_target_context if use_gold_target_context else example.generated_target_context
    has_target_context = target_context is not None and pd.notnull(target_context)
    target_current = example.gold_target_current if use_gold_target_current else example.generated_target_current
    if (
        example.source_context is None
        or not example.source_context
        or not pd.notnull(example.source_context)
        or not pd.notnull(target_current)
    ):
        return None
    target_full = target_context + f"{context_separator} " + target_current if has_target_context else target_current
    use_lang_tag = has_lang_tag(model)
    lang_tag_offset = 1 if use_lang_tag else 0
    if target_tags is None:
        if example.gold_target_current_tagged is None:
            raise ValueError(
                "Provided examples does not have a gold tagged version to infer target tags, and no custom target "
                "tags were provided.\n Target tags are needed to select token identified as context-sensitive and "
                " extract only their scores.\nPlease provide examples containing the gold_target_current_tagged "
                "or pass custom target tags."
            )
        _, target_tags = get_model_cue_target_tags(
            example.gold_target_current_tagged,
            example.gold_target_current if use_gold_target_current else example.generated_target_current,
            model,
            is_generated_untagged=not use_gold_target_current,
            model_type=model_type,
        )
    source_context_tokens = tokenize_subwords(
        example.source_context,
        model,
        model_type=model_type,
        is_target=False,
        special_tokens=["<pad>", "</s>"],
        special_characters=[],
    )
    source_sep_idx = len(source_context_tokens)
    target_context_tokens = None
    if has_target_context:
        target_context_tokens = tokenize_subwords(
            target_context,
            model,
            model_type=model_type,
            is_target=True,
            special_tokens=["<pad>", "</s>"],
            special_characters=[],
        )
        target_sep_idx = len(target_context_tokens)
    target_tags_indices = [i for i, tag in enumerate(target_tags) if tag]
    if not target_tags_indices:
        logger.warning(f"No target tags found for example {curr_idx}.")

    out_df = None
    for cti_idx in target_tags_indices:
        curr_idx_out_df = None
        output_current_contrast, aligns, pos_start = prepare_cci_params(
            model=model,
            has_output_context=has_target_context,
            output_context=target_context,
            output_current=target_current,
            input_current=example.source_current,
            impute_with_contextless_output=impute_with_contextless_output,
            ctx_break=context_separator,
            cti_tok_idx=cti_idx - lang_tag_offset,
            model_use_ctx_break=True,
            model_has_lang_tag=has_lang_tag(model),
            force_context_aware_output_prefix=force_context_aware_output_prefix,
            gen_kwargs=gen_kwargs,
        )
        for attribution_method in attribution_methods:
            model.setup(attribution_method=attribution_method)
            curr_attributed_fns = attributed_fns
            if not model.attribution_method.use_predicted_target:
                curr_attributed_fns = ["default"]
            for attributed_fn in curr_attributed_fns:
                attribute_kwargs = {}
                if attributed_fn != "default":
                    attribute_kwargs["attributed_fn"] = attributed_fn
                    attribute_kwargs["contrast_sources"] = example.source_current
                    attribute_kwargs["contrast_targets"] = output_current_contrast
                    attribute_kwargs["contrast_targets_alignments"] = aligns
                cci_out = model.attribute(
                    example.source_full,
                    target_full,
                    attribute_target=True,
                    show_progress=False,
                    attr_pos_start=pos_start,
                    attr_pos_end=pos_start + 1,
                    **attribute_kwargs,
                )
                if attributed_fn == "default":
                    aggr_out = cci_out[0].aggregate(normalize=False)
                else:
                    aggr_out = cci_out[0].aggregate("sum", normalize=False)

                # Extract context attribution for every context-sensitive token
                source_context_scores = aggr_out.source_attributions[
                    lang_tag_offset : lang_tag_offset + source_sep_idx, 0
                ].tolist()
                curr_scores_df = pd.DataFrame(
                    {
                        "example_idx": curr_idx,
                        "token_idx": [i for i, _ in enumerate(source_context_tokens)],
                        "side": "S",
                        "cti_idx": cti_idx,
                        "token": source_context_tokens,
                        f"{attribution_method}_{attributed_fn}": source_context_scores,
                    }
                )
                if include_per_unit_scores and attributed_fn == "default":
                    unaggr_source_context_scores = cci_out[0].source_attributions[
                        lang_tag_offset : lang_tag_offset + source_sep_idx, 0, ...
                    ]
                    source_scores_size = list(unaggr_source_context_scores.size())
                    unit_sizes = dict(zip(units_names, source_scores_size[::-1][: len(units_names)]))
                    unit_combinations = product(*[range(unit_sizes[unit]) for unit in units_names])
                    for unit_combination in unit_combinations:
                        unit_combination = tuple(reversed(unit_combination))
                        name_id = "_".join([f"{unit}{idx}" for unit, idx in zip(unit_combination, units_names)])
                        curr_scores_df[f"{attribution_method}_{name_id}"] = unaggr_source_context_scores[
                            (...,) + unit_combination
                        ]
                if has_target_context:
                    target_context_scores = aggr_out.target_attributions[
                        lang_tag_offset : lang_tag_offset + target_sep_idx, 0
                    ].tolist()
                    curr_scores_target_df = pd.DataFrame(
                        {
                            "example_idx": curr_idx,
                            "token_idx": [i for i, _ in enumerate(target_context_tokens)],
                            "side": "T",
                            "cti_idx": cti_idx,
                            "token": target_context_tokens,
                            f"{attribution_method}_{attributed_fn}": target_context_scores,
                        }
                    )
                    if include_per_unit_scores and attributed_fn == "default":
                        unaggr_target_context_scores = cci_out[0].target_attributions[
                            lang_tag_offset : lang_tag_offset + target_sep_idx, 0, ...
                        ]
                        target_scores_size = list(unaggr_target_context_scores.size())
                        unit_sizes = dict(zip(units_names, target_scores_size[::-1][: len(units_names)]))
                        unit_combinations = product(*[range(unit_sizes[unit]) for unit in units_names])
                        for unit_combination in unit_combinations:
                            unit_combination = tuple(reversed(unit_combination))
                            name_id = "_".join([f"{unit}{idx}" for unit, idx in zip(unit_combination, units_names)])
                            curr_scores_target_df[f"{attribution_method}_{name_id}"] = unaggr_target_context_scores[
                                (...,) + unit_combination
                            ]
                    curr_scores_df = pd.concat([curr_scores_df, curr_scores_target_df], ignore_index=True)
                if curr_idx_out_df is None:
                    curr_idx_out_df = curr_scores_df
                else:
                    curr_idx_out_df = pd.merge(
                        curr_idx_out_df, curr_scores_df, on=["example_idx", "token_idx", "token", "side", "cti_idx"]
                    )
        if out_df is None:
            out_df = curr_idx_out_df
        else:
            out_df = pd.concat([out_df, curr_idx_out_df], ignore_index=True)
    return out_df
