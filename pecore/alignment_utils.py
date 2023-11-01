import re
from typing import Any, List, Optional, Tuple

from stanza import Pipeline

from inseq import AttributionModel
from inseq.utils.alignment_utils import align_tokenizations, compute_word_aligns

from .enums import ModelTypeEnum
from .model_utils import get_model_attribute, has_lang_tag


def tokenize_words(text: str, is_tagged: bool = False) -> List[str]:
    """Tokenizes a string of words splitting on word boundaries and spaces, optionally with tags."""
    pattern_nonspace = r"(<p>|</p>|<hon>|<hoff>|\S+)" if is_tagged else r"(\S+)"
    pattern_word = r"(<p>|</p>|<hon>|<hoff>|\w+)" if is_tagged else r"(\w+)"
    return [x for nonspace in re.split(pattern_nonspace, text) for x in re.split(pattern_word, nonspace) if x.strip()]


def tokenize_subwords(
    text: str,
    model: AttributionModel,
    special_characters: List[str] = ["▁"],
    special_tokens: List[str] = ["<pad>", "</s>"],
    model_type: Optional[ModelTypeEnum] = None,
    is_target: bool = True,
) -> List[str]:
    """Tokenizes a string of words into subwords using the given model's tokenizer."""
    out = model.encode(text, as_targets=is_target)
    tokens = out.input_tokens[0]
    for char in special_characters:
        tokens = [t.strip(char) for t in tokens]
    if has_lang_tag(model):
        if model_type is None:
            raise ValueError("Model type must be specified if the model has a language tag.")
        lang_tags = get_model_attribute(model_type, "lang_map").values()
        special_tokens += list(lang_tags)
    return [t for t in tokens if t not in special_tokens]


def get_match_from_contrastive_pair(
    ref_text: str,
    contrast_ref_text: str,
    pred_text: str,
) -> List[int]:
    """Returns a list of 0s and 1s, where 0 means that the word is not in the MT output and 1 means that it is."""
    ref_tok = re.findall(r"\w+\b", ref_text)
    contrast_ref_tok = re.findall(r"\w+\b", contrast_ref_text)
    if not isinstance(pred_text, str):
        return [0]
    pred_tok = [x.lower() for x in re.findall(r"\w+\b", pred_text)]
    keywords = [ref.lower() for ref, con in zip(ref_tok, contrast_ref_tok) if ref != con]
    out = []
    for kw in keywords:
        if kw not in pred_tok:
            out += [0]
        else:
            out += [1]
            pred_tok.remove(kw)
    return out


def get_match_from_tagged_contrastive_pair(
    tagged_ref_text: str,
    tagged_contrast_ref_text: str,
    pred_text: str,
    fulltext_match: bool = False,
) -> List[int]:
    """Returns a list of 0s and 1s, where 0 means that the word is not in the MT output and 1 means that it is."""
    tag_content_pattern = r"<\w+>(.+?)</?\w+>"
    tagged_ref_tok = re.findall(tag_content_pattern, tagged_ref_text)
    tagged_contrast_ref_tok = re.findall(tag_content_pattern, tagged_contrast_ref_text)
    if not isinstance(pred_text, str):
        return [0]
    keywords = [ref.lower() for ref in tagged_ref_tok if ref not in tagged_contrast_ref_tok]
    out = []
    if fulltext_match:
        pred_text = pred_text.lower()
        for kw in keywords:
            if kw not in pred_text:
                out += [0]
            else:
                out += [1]
                pred_text = pred_text.replace(kw, "", 1)
    else:
        pred_tok = [x.lower() for x in re.findall(r"\w+\b", pred_text)]
        for kw in keywords:
            if kw not in pred_tok:
                out += [0]
            else:
                out += [1]
                pred_tok.remove(kw)
    return out


def get_tokens_with_cue_target_tags(
    txt_tag: str, txt_clean: Optional[str] = None, tags: List[str] = ["<p>", "<hon>", "</p>", "<hoff>"]
) -> Tuple[List[str], List[int], List[int]]:
    """Given a tagged and untagged version of the same text, returns a tuple containing:

    - The word-level tokens of the untagged text
    - A list of 0s and 1s, where 1 means that the respective i-th word is a cue word and 0 means that it is not
    - A list of 0s and 1s, where 1 means that the respective i-th word is a target word and 0 means that it is not
    """
    if txt_clean is None:
        for tag in tags:
            if txt_clean is None:
                txt_clean = txt_tag.replace(tag, "")
            else:
                txt_clean = txt_clean.replace(tag, "")
    untagged_toks = tokenize_words(txt_clean)
    tagged_toks = tokenize_words(txt_tag, is_tagged=True)
    tag_idx, untag_idx = 0, 0
    cue_tags = [0 for _ in range(len(untagged_toks))]
    target_tags = [0 for _ in range(len(untagged_toks))]
    is_cue = False
    is_target = False
    while tag_idx < len(tagged_toks) and untag_idx < len(untagged_toks):
        if tagged_toks[tag_idx] == untagged_toks[untag_idx]:
            if is_cue:
                cue_tags[untag_idx] = 1
            elif is_target:
                target_tags[untag_idx] = 1
            tag_idx += 1
            untag_idx += 1
        elif tagged_toks[tag_idx] in ["<p>", "<hon>"]:
            if tagged_toks[tag_idx] == "<p>":
                is_target = True
            elif tagged_toks[tag_idx] == "<hon>":
                is_cue = True
            tag_idx += 1
        elif tagged_toks[tag_idx] in ["</p>", "<hoff>"]:
            if tagged_toks[tag_idx] == "</p>":
                is_target = False
            elif tagged_toks[tag_idx] == "<hoff>":
                is_cue = False
            tag_idx += 1
        else:
            print(tagged_toks[tag_idx], untagged_toks[untag_idx])
            raise ValueError(f"Something went wrong\nTagged:{tagged_toks}\nUntagged:{untagged_toks}")
    return untagged_toks, cue_tags, target_tags


def propagate_tags(
    tok_a: List[str], tok_b_tags: List[Any], a_to_b_alignments: List[Tuple[int, int]], default_val: Any = 0
) -> List[int]:
    tok_a_tags = [default_val for _ in range(len(tok_a))]
    for tok_a_idx, tok_b_idx in a_to_b_alignments:
        tok_a_tags[tok_a_idx] = tok_b_tags[tok_b_idx]
    return tok_a_tags


def get_model_cue_target_tags(
    tagged: str,
    untagged: str,
    model: AttributionModel,
    is_generated_untagged: bool = True,
    model_type: Optional[ModelTypeEnum] = None,
    is_target: bool = True,
    is_current: bool = True,
    add_lang_tag: bool = True,
) -> Tuple[List[int], List[int]]:
    subword_tokenized = tokenize_subwords(untagged, model, model_type=model_type, is_target=is_target)
    # Get cue and target tags on the gold word-tokenized text
    word_tokenized, word_cue_tags, word_target_tags = get_tokens_with_cue_target_tags(
        tagged, None if is_generated_untagged else untagged
    )
    if is_generated_untagged:
        gen_word_tokenized = tokenize_words(untagged)
        # Align the word-tokenized model generation to the word-tokenized gold text
        gen_to_gold_word_alignments = compute_word_aligns(gen_word_tokenized, word_tokenized).alignments
        # Tags on the model-generated word level translation
        word_cue_tags = propagate_tags(gen_word_tokenized, word_cue_tags, gen_to_gold_word_alignments)
        word_target_tags = propagate_tags(gen_word_tokenized, word_target_tags, gen_to_gold_word_alignments)
        word_tokenized = gen_word_tokenized
    # Align the subword- and word-tokenized sequences
    alignments = align_tokenizations(subword_tokenized, word_tokenized).alignments
    # Propagate word-level tags on model generation to subword level.
    subword_cue_tags = propagate_tags(subword_tokenized, word_cue_tags, alignments)
    subword_target_tags = propagate_tags(subword_tokenized, word_target_tags, alignments)
    # Add </s> token tag
    if is_target and is_current:
        subword_cue_tags += [0]
        subword_target_tags += [0]
    if has_lang_tag(model) and add_lang_tag:
        subword_cue_tags = [0] + subword_cue_tags
        subword_target_tags = [0] + subword_target_tags
    return subword_cue_tags, subword_target_tags


def get_model_lang_feats(
    sent: str,
    pipeline: Pipeline,
    model: AttributionModel,
    model_type: Optional[ModelTypeEnum] = None,
    is_target: bool = True,
    is_current: bool = True,
    add_lang_tag: bool = True,
) -> Tuple[List[int], List[int]]:
    doc = pipeline(sent)
    word_tokenized = [token.text for sent in doc.sentences for token in sent.tokens]
    word_pos_tags = ["+".join(word.upos for word in token.words) for sent in doc.sentences for token in sent.tokens]
    word_feats_tags = [
        "+".join(word.feats if word.feats else "_" for word in token.words)
        for sent in doc.sentences
        for token in sent.tokens
    ]
    subword_tokenized = tokenize_subwords(sent, model, model_type=model_type, is_target=is_target)
    alignments = align_tokenizations(subword_tokenized, word_tokenized).alignments
    subword_pos_tags = propagate_tags(subword_tokenized, word_pos_tags, alignments, default_val="X")
    subword_feats_tags = propagate_tags(subword_tokenized, word_feats_tags, alignments, default_val="_")
    if is_target and is_current:
        subword_pos_tags += ["EOS"]
        subword_feats_tags += ["_"]
    if has_lang_tag(model) and add_lang_tag:
        subword_pos_tags = ["LANG"] + subword_pos_tags
        subword_feats_tags = ["_"] + subword_feats_tags
    return subword_pos_tags, subword_feats_tags
