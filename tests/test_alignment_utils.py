from typing import Dict

from pecore.alignment_utils import get_model_cue_target_tags, tokenize_subwords
from pytest import fixture

import inseq
from inseq.models import HuggingfaceEncoderDecoderModel


@fixture(scope="session")
def marian_model() -> HuggingfaceEncoderDecoderModel:
    return inseq.load_model("Helsinki-NLP/opus-mt-en-it", "dummy")


@fixture(scope="session")
def mbart_model() -> HuggingfaceEncoderDecoderModel:
    mbart = inseq.load_model("facebook/mbart-large-50-one-to-many-mmt", "dummy")
    mbart.tokenizer.src_lang = "en_XX"
    mbart.tokenizer.tgt_lang = "fr_XX"
    return mbart


@fixture(scope="session")
def target_only_gold() -> Dict[str, str]:
    return {
        "tgt": "Elles rencontrent des mecs sympas, elles peuvent même avoir plus.",
        "tgt_tag": "<p>Elles</p> rencontrent des mecs sympas, elles peuvent même avoir plus.",
        "cues": [],
        "targets": ["Elles"],
    }


@fixture(scope="session")
def cue_target_gold() -> Dict[str, str]:
    return {
        "tgt": "J'ai donné mon opinion plutôt cet après-midi, et elle n'a toujours pas changé.",
        "tgt_tag": "J'ai déjà donné mon  <hon>opinion<hoff>  cet après-midi ; <p>elle</p> n'a pas changé.",
        "cues": ["opinion"],
        "targets": ["elle"],
    }


def check_conformity(out, cue_tags, target_tags, cues=[], targets=[]):
    assert sum(cue_tags) == len(cues)
    assert sum(target_tags) == len(targets)
    assert len(cue_tags) == len(out.input_tokens[0])
    assert len(target_tags) == len(out.input_tokens[0])
    for cue in cues:
        c_idx = [t for t, tok in enumerate(out.input_tokens[0]) if cue == tok]
        assert len(c_idx) > 0
        c_idx = c_idx[0]
        assert cue_tags[c_idx] == 1
    for target in targets:
        t_idx = [t for t, tok in enumerate(out.input_tokens[0]) if target == tok]
        assert len(t_idx) > 0
        t_idx = t_idx[0]
        assert target_tags[t_idx] == 1


def extract_tags_and_check(model, example, is_generated=True, model_type=None):
    curr_example = example.copy()
    tgt = curr_example.pop("tgt")
    tgt_tag = curr_example.pop("tgt_tag")
    cue_tags, target_tags = get_model_cue_target_tags(
        tgt_tag, tgt, model, is_generated_untagged=is_generated, model_type=model_type
    )
    out = model.encode(tgt, as_targets=True, add_bos_token=False)
    if len(curr_example["cues"]) > 0:
        new_cues = []
        for cue in curr_example["cues"]:
            new_cues += tokenize_subwords(cue, model, special_characters=[], model_type=model_type)
        curr_example["cues"] = new_cues
    if len(curr_example["targets"]) > 0:
        new_targets = []
        for target in curr_example["targets"]:
            new_targets += tokenize_subwords(target, model, special_characters=[], model_type=model_type)
        curr_example["targets"] = new_targets
    check_conformity(out, cue_tags, target_tags, **curr_example)


def test_get_model_cue_target_tags_gold_target(marian_model, mbart_model, target_only_gold, cue_target_gold):
    extract_tags_and_check(marian_model, target_only_gold, is_generated=False)
    extract_tags_and_check(mbart_model, target_only_gold, is_generated=False, model_type="mbart50-1toM")
    extract_tags_and_check(marian_model, cue_target_gold)
    extract_tags_and_check(mbart_model, cue_target_gold, model_type="mbart50-1toM")
