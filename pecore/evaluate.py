import logging
import os
import re
from typing import List, Tuple

from comet import download_model, load_from_checkpoint
from sacrebleu.metrics import BLEU
from tqdm import tqdm

from .translate import DATASETS

model_path = download_model("Unbabel/wmt22-comet-da")
model = load_from_checkpoint(model_path)

loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    logger.setLevel(logging.WARNING)


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


def evaluate(
    cwd=0,
    ctx=0,
    model_type="",
    dataset="flores",
    src_lang="eng",
    tgt_lang="fra",
    metric="bleu",
    use_context=True,
    model_name=None,
    use_target_context=False,
):
    if model_name is None:
        model_id = f"{model_type}-ctx{ctx}-cwd{cwd}"
    else:
        model_id = model_type
    if dataset == "flores":
        src = [f"sentence_{src_lang[:3]}"]
        refs = DATASETS[dataset][f"sentence_{tgt_lang[:3]}"]
    elif dataset == "iwslt17":
        src = [ex[src_lang[:2]] for ex in DATASETS[dataset]["translation"]]
        refs = [ex[tgt_lang[:2]] for ex in DATASETS[dataset]["translation"]]
    elif dataset == "scat":
        src = DATASETS[dataset][src_lang[:2]]
        refs = DATASETS[dataset][tgt_lang[:2]]
    base_path = "translations/translations" if use_context else "translations/translations_noctx"
    with open(os.path.join(base_path, f"{dataset}-{model_id}{'' if use_context else '-noctx'}.txt")) as f:
        sys = f.readlines()
    if use_target_context:
        # Remove target context for evaluation
        sys = [s.split("<brk>")[1].strip() if "<brk>" in s else s for s in sys]
    if metric == "comet":
        comet_out = model.predict(
            [{"src": s, "mt": m, "ref": r} for s, m, r in zip(src, sys, refs)], batch_size=8, gpus=1
        )
        print(dataset, f"{model_id}{'' if use_context else '-noctx'}", "COMET", comet_out.system_score)
    if metric == "bleu":
        bleu = BLEU()
        print(dataset, f"{model_id}{'' if use_context else '-noctx'}", bleu.corpus_score(sys, [refs]))
    if metric == "accuracy":
        if dataset != "scat":
            raise ValueError("Only scat dataset supports accuracy metric")
        tot_keywords, tot_correct = 0, 0
        for curr_ref, curr_ref_contrast, curr_mt in tqdm(
            zip(refs, DATASETS[dataset][f"contrast_{tgt_lang[:2]}"], sys), desc="Aligned accuracy", total=len(refs)
        ):
            matches = get_aligned_gender_annotations(curr_ref, curr_ref_contrast, curr_mt)
            tot_keywords += len(matches)
            tot_correct += len([x for x in matches if x == 1])
        print(
            dataset,
            f"{model_id}{'' if use_context else '-noctx'}",
            "align_match_acc",
            round(tot_correct / tot_keywords, 4),
        )
