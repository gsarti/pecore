import re
from typing import List, Tuple

from simalign import SentenceAligner

aligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="mai")


def tokenize(text: str, is_tagged: bool = False):
    pattern_nonspace = r"(<p>|</p>|<hon>|<hoff>|\S+)" if is_tagged else r"(\S+)"
    pattern_word = r"(<p>|</p>|<hon>|<hoff>|\w+)" if is_tagged else r"(\w+)"
    return [x for nonspace in re.split(pattern_nonspace, text) for x in re.split(pattern_word, nonspace) if x.strip()]


def tokenize_model(text: str, model):
    out = model.encode(text, as_targets=True)
    return [x.replace("▁", "") for x in out.input_tokens[0] if x not in ["<pad>", "</s>", "fr_XX"]]


def get_aligned_gender_annotations(ref_text, contrast_ref_text, mt_text) -> List[Tuple[str, str]]:
    """Returns a list of 0s and 1s, where 0 means that the word is not in the MT output and 1 means that it is."""
    ref_tok = re.findall(r"\w+\b", ref_text)
    contrast_ref_tok = re.findall(r"\w+\b", contrast_ref_text)
    if not isinstance(mt_text, str):
        return [0]
    mt_tok = [x.lower() for x in re.findall(r"\w+\b", mt_text)]
    keywords = [ref for ref, con in zip(ref_tok, contrast_ref_tok) if ref != con]
    out = []
    for kw in keywords:
        if kw.lower() not in mt_tok:
            out += [0]
        else:
            out += [1]
            mt_tok.remove(kw.lower())
    return out


def get_tokens_with_cue_target_tags(txt_tag: str, txt_clean: str):
    untagged_toks = tokenize(txt_clean)
    tagged_toks = tokenize(txt_tag, is_tagged=True)
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


def get_subword_alignments(src: str, tgt: str) -> List[Tuple[int, int]]:
    """Aligns tokens of two whitespace-tokenized strings having the same contents,
    but differing in tokenization.
    The output is a sequence in the format "0-0 1-1 2-3 3-2 ..." corresponding to indices of
    aligned tokens between src and tgt
    """
    assert "".join(src.split(" ")) == "".join(
        tgt.split(" ")
    ), f"SRC: {''.join(src.split())}\nTGT: {''.join(tgt.split())}\n"
    out = []
    src_idx = 0
    tgt_idx = 0
    # Splitting on single space ensures that "_" tokens are not lost and alignments are preserved.
    src_tok = src.strip().split(" ")
    tgt_tok = tgt.strip().split(" ")
    while src_idx < len(src_tok):
        curr_src_tok = src_tok[src_idx]
        curr_tgt_tok = tgt_tok[tgt_idx]
        if curr_src_tok == curr_tgt_tok:
            out.append(f"{src_idx}-{tgt_idx}")
            src_idx += 1
            tgt_idx += 1
        elif curr_src_tok in curr_tgt_tok:
            out.append(f"{src_idx}-{tgt_idx}")
            tgt_tok[tgt_idx] = tgt_tok[tgt_idx].replace(curr_src_tok, "", 1)
            src_idx += 1
        elif curr_tgt_tok in curr_src_tok:
            out.append(f"{src_idx}-{tgt_idx}")
            src_tok[src_idx] = src_tok[src_idx].replace(curr_tgt_tok, "", 1)
            tgt_idx += 1
        else:
            raise ValueError(f"ERR: {curr_src_tok} =!= {curr_tgt_tok}")
    out = " ".join(out)
    return [tuple(int(x) for x in pair.split("-")) for pair in out.split()]


def propagate_tags(tok_tgt, tags, alignments):
    model_tok_cue_tags = [0 for _ in range(len(tok_tgt))]
    for tok_idx, word_idx in alignments:
        if tags[word_idx] == 1:
            model_tok_cue_tags[tok_idx] = 1
    return model_tok_cue_tags


def get_model_cue_target_tags(tagged, untagged, model, ex=None, has_lang_tag=False):
    if ex is None:
        model_tokenized = tokenize_model(untagged, model)
        untagged_toks, cue_tags, target_tags = get_tokens_with_cue_target_tags(tagged, untagged)
        try:
            alignments = get_subword_alignments(" ".join(model_tokenized), " ".join(untagged_toks))
        except AssertionError as e:
            raise ValueError(model_tokenized, untagged_toks) from e
        cue_tags = propagate_tags(model_tokenized, cue_tags, alignments)
        target_tags = propagate_tags(model_tokenized, target_tags, alignments)
        return cue_tags, target_tags
    else:
        word_tok_ctx_gen = tokenize(ex["fr"])
        sub_tok_ctx_gen = tokenize_model(ex["fr"], model)
        # Get cue and target tags on the gold word-tokenized text
        tok_gold_ref, gold_word_cue_tags, gold_word_target_tags = get_tokens_with_cue_target_tags(
            ex["orig_fr_with_tags"], ex["tgt_fr"]
        )
        # Align the word-tokenized model generation to the word-tokenized gold text
        ctx_gen_to_gold_ref_alignments = aligner.get_word_aligns(word_tok_ctx_gen, tok_gold_ref)["itermax"]
        # Tags on the model-generated word level translation
        ctx_gen_word_cue_tags = propagate_tags(word_tok_ctx_gen, gold_word_cue_tags, ctx_gen_to_gold_ref_alignments)
        ctx_gen_word_target_tags = propagate_tags(
            word_tok_ctx_gen, gold_word_target_tags, ctx_gen_to_gold_ref_alignments
        )
        # Align the subword- and word-tokenized model generations
        try:
            sub_to_word_ctx_gen_alignments = get_subword_alignments(
                " ".join(sub_tok_ctx_gen), " ".join(word_tok_ctx_gen)
            )
        except AssertionError as e:
            raise ValueError(sub_tok_ctx_gen, word_tok_ctx_gen) from e
        # Propagate word-level tags on model generation to subword level.
        cue_tags = propagate_tags(sub_tok_ctx_gen, ctx_gen_word_cue_tags, sub_to_word_ctx_gen_alignments)
        target_tags = propagate_tags(sub_tok_ctx_gen, ctx_gen_word_target_tags, sub_to_word_ctx_gen_alignments)
    # Add </s> token tag
    cue_tags += [0]
    target_tags += [0]
    if has_lang_tag:
        cue_tags = [0] + cue_tags
        target_tags = [0] + target_tags
    return cue_tags, target_tags
