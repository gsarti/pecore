from typing import List, Optional, Tuple

import inseq
import torch
from inseq.attr.step_functions import StepFunctionArgs, _get_contrast_output
from inseq.data import FeatureAttributionInput
from inseq.utils import logits_kl_divergence


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


def base_attribute_fn(ex: Dict[str, str], model: AttributionModel, idx: int) -> pd.DataFrame:
    out = model.attribute(
        ex["src_en"],
        ex["tgt_fr"],
        attribute_target=True,
        step_scores=["probability", "contrast_prob", "pcxmi", "kl_divergence"],
        contrast_sources=ex["src_en_ctx"],
        contrast_target_prefixes=ex["tgt_fr_ctx"],
        show_progress=False,
    )
    df = pd.DataFrame(out.get_scores_dicts()[0]["step_scores"])
    df = df.transpose().reset_index().rename(columns={"level_0": "token_idx", "level_1": "token"})
    df.insert(0, "example_idx", idx)
    return df


def top_p_attribute_fn(ex: Dict[str, str], model: AttributionModel, idx: int) -> pd.DataFrame:
    overall_df = None
    for top_p in [0.1, 0.3, 0.5, 0.7, 0.9]:
        out = model.attribute(
            ex["src_en"],
            ex["tgt_fr"],
            attribute_target=True,
            step_scores=["kl_divergence", "top_p_size"],
            contrast_sources=ex["src_en_ctx"],
            contrast_target_prefixes=ex["tgt_fr_ctx"],
            show_progress=False,
            top_p=top_p,
        )
        df = pd.DataFrame(out.get_scores_dicts()[0]["step_scores"])
        df = df.transpose().reset_index().rename(columns={"level_0": "token_idx", "level_1": "token"})
        df.insert(0, "example_idx", idx)
        if overall_df is None:
            overall_df = df.rename(
                columns={"kl_divergence": f"kl_div_{int(top_p * 100)}", "top_p_size": f"top_p_size_{int(top_p * 100)}"}
            )
        else:
            overall_df[f"kl_div_{int(top_p * 100)}"] = df["kl_divergence"]
            overall_df[f"top_p_size_{int(top_p * 100)}"] = df["top_p_size"]
    return overall_df


def logit_lens_attribute_fn(ex: Dict[str, str], model: AttributionModel, idx: int) -> pd.DataFrame:
    out = model.attribute(
        ex["src_en"],
        ex["tgt_fr"],
        attribute_target=True,
        step_scores=["kl_div_per_layer"],
        contrast_sources=ex["src_en_ctx"],
        contrast_target_prefixes=ex["tgt_fr_ctx"],
        show_progress=False,
    )
    for layer_idx in range(out[0].step_scores["kl_div_per_layer"].shape[0]):
        out[0].step_scores[f"kl_div_l{layer_idx}"] = out[0].step_scores["kl_div_per_layer"][layer_idx, :]
    del out[0].step_scores["kl_div_per_layer"]
    df = pd.DataFrame(out.get_scores_dicts()[0]["step_scores"])
    df = df.transpose().reset_index().rename(columns={"level_0": "token_idx", "level_1": "token"})
    df.insert(0, "example_idx", idx)
    return df


def attribute_contrast(ex: Dict[str, str], model: AttributionModel, idx: int) -> FeatureAttributionOutput:
    has_target_context = ex["tgt_fr_ctx"] is not None and pd.notnull(ex["tgt_fr_ctx"])
    # Handle missing source context
    full_src = ex["src_en_ctx"]
    if full_src.startswith("<brk>") or not isinstance(ex["tgt_fr"], str):
        print(f"Skipping example {idx}")
        return None
    full_tgt = ex["tgt_fr_ctx"] + " " + ex["tgt_fr"].strip() if has_target_context else ex["tgt_fr"].strip()
    tgt_fr_ctx_tokens = model.encode(ex["tgt_fr_ctx"], as_targets=True).input_tokens[0]
    offset = len(tgt_fr_ctx_tokens) - 1 if has_target_context else 0  # pad
    if tgt_fr_ctx_tokens[1] == "fr_XX" and has_target_context:
        offset -= 1
    curr_len = len(model.encode(ex["tgt_fr"], as_targets=True).input_tokens[0]) - 1  # pad
    out = model.attribute(
        full_src,
        full_tgt,
        attribute_target=True,
        show_progress=False,
        attr_pos_start=offset if has_target_context else None,
        attributed_fn="contrast_prob_diff",
        contrast_sources=ex["src_en"],
        contrast_targets=ex["tgt_fr"].strip(),
        contrast_targets_alignments=[
            (idx_full, idx_curr)
            for idx_curr, idx_full in enumerate(range(offset, offset + curr_len), start=1 if has_target_context else 0)
        ],
    )
    return out


def input_contributions_attribute_fn(ex: Dict[str, str], model: AttributionModel, idx: int) -> pd.DataFrame:
    has_target_context = ex["tgt_fr_ctx"] is not None and pd.notnull(ex["tgt_fr_ctx"])
    out = attribute_contrast(ex, model, idx)
    has_lang_tag = out[0].source[0].token == "en_XX" and out[0].target[0].token == "fr_XX"
    aggr_args = {}
    src_brk_idx = [t.token for t in out[0].source].index("<brk>")
    lang_tag_offset = 1 if has_lang_tag else 0
    aggr_args["source_spans"] = [(lang_tag_offset, src_brk_idx), (src_brk_idx + 1, len(out[0].source) - 1)]
    if has_target_context:
        special_tok = "fr_XX → <brk>" if has_lang_tag else "<brk>"
        tgt_brk_idx = [t.token for t in out[0].target].index(special_tok)
        aggr_args["target_spans"] = [(lang_tag_offset, tgt_brk_idx)]
    aggr_out = out.aggregate("spans", **aggr_args).aggregate()
    assert aggr_out[0].source_attributions.size(0) == 4 + lang_tag_offset, (
        f"Expected {4 + lang_tag_offset} source tokens but found {aggr_out[0].source_attributions.size(0)} "
        f"instead: {aggr_out[0].source}"
    )
    if has_lang_tag:
        aggr_out[0].step_scores["src_langtag_attr"] = aggr_out[0].source_attributions[0, :]
    aggr_out[0].step_scores["src_ctx_attr"] = aggr_out[0].source_attributions[0 + lang_tag_offset, :]
    aggr_out[0].step_scores["src_brk_attr"] = aggr_out[0].source_attributions[1 + lang_tag_offset, :]
    aggr_out[0].step_scores["src_curr_attr"] = aggr_out[0].source_attributions[2 + lang_tag_offset, :]
    aggr_out[0].step_scores["src_eos_attr"] = aggr_out[0].source_attributions[3 + lang_tag_offset, :]
    tgt_curr_start_idx = 0
    if has_lang_tag:
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
        df["token_idx"] = (i for i in range(len(aggr_out[0].step_scores["src_ctx_attr"])))
    df.insert(0, "example_idx", idx)
    return df
