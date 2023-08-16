from collections import Counter, OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import inseq
import stanza
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm

from .enums import DatasetEnum

nlp = stanza.Pipeline(lang="en", processors="tokenize", download_method=None)

DOC_FIELD_MAP = {
    "flores": "URL",
    "iwslt17": "doc_id",
}


def load_mt_dataset(dataset_name: str, src_lang: Optional[str] = None, tgt_lang: Optional[str] = None) -> Dataset:
    if dataset_name == DatasetEnum.FLORES:
        return load_dataset("gsarti/flores_101", "all", split="devtest")
    elif dataset_name == DatasetEnum.IWSLT:
        return load_dataset("gsarti/iwslt2017_context", f"iwslt2017-{src_lang[:2]}-{tgt_lang[:2]}", split="test")
    elif dataset_name == DatasetEnum.SCAT:
        return load_dataset("inseq/scat", split="test")
    else:
        raise ValueError(f"Not available: {dataset_name}")


def get_src_ref_sentences(
    dataset_name: str,
    src_lang: str,
    tgt_lang: str,
    dataset: Optional[Dataset] = None,
) -> Tuple[List[str], List[str]]:
    if dataset is None:
        dataset = load_mt_dataset(dataset_name, src_lang, tgt_lang)
    if dataset_name == DatasetEnum.FLORES:
        src = dataset[f"sentence_{src_lang[:3]}"]
        ref = dataset[f"sentence_{tgt_lang[:3]}"]
    elif dataset_name == DatasetEnum.IWSLT:
        src = [ex[src_lang[:2]] for ex in dataset["translation"]]
        ref = [ex[tgt_lang[:2]] for ex in dataset["translation"]]
    elif dataset_name == DatasetEnum.SCAT:
        src = dataset[src_lang[:2]]
        ref = dataset[tgt_lang[:2]]
    else:
        raise ValueError(f"Not available: {dataset_name}")
    return src, ref


# Used in dataset preprocessing to sample the number of context sentences to include
class OrderedCounter(Counter, OrderedDict):
    "Counter that remembers the order elements are first encountered"

    def __repr__(self):
        return f"{self.__class__.__name__}({OrderedDict(self)!r})"

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


def get_preprocess_dataset_fn(context_size: int, dataset: str = "flores", src_lang: str = "eng"):
    """Returns a preprocessing function for the specified dataset and context size to build context-aware examples."""

    def preprocess_dataset_seq(examples: Dict[str, List[Any]]):
        """Builds context-aware examples for datasets with sequential examples (e.g. IWSLT, Flores)."""
        if dataset == DatasetEnum.FLORES:
            inputs = examples[f"sentence_{src_lang[:3]}"]
        elif dataset == DatasetEnum.IWSLT:
            inputs = [ex[src_lang[:2]] for ex in examples["translation"]]
        else:
            raise ValueError(f"Not available: {dataset}")
        doc_field = examples[DOC_FIELD_MAP[dataset]]
        n_previous = [i for _, v in OrderedCounter(doc_field).items() for i in range(v)]
        n_contexts = [min(context_size, n_previous[idx]) for idx in range(len(inputs))]
        context_inputs = []
        for idx in range(len(inputs)):
            if n_contexts[idx] > 0:
                ctx = " ".join(inputs[idx - n_contexts[idx] : idx])
                context_inputs.append(f"{ctx}<brk> {inputs[idx]}")
            else:
                context_inputs.append(inputs[idx])
        return {"sentence": context_inputs}

    def preprocess_dataset_merged(examples: Dict[str, List[Any]]):
        """Builds context-aware examples for datasets with context available in the same example (e.g. SCAT)."""
        if dataset == DatasetEnum.SCAT:
            inputs = examples[src_lang[:2]]
            contexts = examples[f"context_{src_lang[:2]}"]
        else:
            raise ValueError(f"Not available: {dataset}")
        context_inputs = []
        for idx in range(len(inputs)):
            if context_size > 0:
                context_inputs.append(f"{contexts[idx]}<brk> {inputs[idx]}")
            else:
                context_inputs.append(inputs[idx])
        return {"sentence": context_inputs}

    if dataset in [DatasetEnum.FLORES, DatasetEnum.IWSLT]:
        return preprocess_dataset_seq
    elif dataset in [DatasetEnum.SCAT]:
        return preprocess_dataset_merged


def get_formatted_examples(
    dataset,
    model_name: str = None,
    force_gen: bool = False,
    has_context: bool = True,
    has_lang_tag: bool = False,
    has_target_context: bool = False,
    start_idx: int = None,
    max_idx: int = None,
) -> List[Dict[str, str]]:
    if max_idx is None:
        max_idx = len(dataset)
    if start_idx is None:
        start_idx = 0
    model = inseq.load_model(model_name, "saliency")
    generate_kwargs = {}
    if has_lang_tag:
        model.tokenizer.src_lang = "en_XX"
        model.tokenizer.tgt_lang = "fr_XX"
        generate_kwargs["forced_bos_token_id"] = model.tokenizer.lang_code_to_id["fr_XX"]
    examples = []
    for idx, ex in tqdm(enumerate(dataset), total=max_idx):
        if idx < start_idx:
            continue
        if max_idx is not None and idx >= max_idx:
            break
        if not force_gen:
            ctx_tgt = None
            if has_context:
                contrast_sources = ex["context_en"] + "<brk> " + ex["en"]
            else:
                contrast_sources = ex["context_en"] + " " + ex["en"]
                ctx_tgt = " ".join(
                    [
                        model.generate(s.text, max_new_tokens=128, **generate_kwargs)[0]
                        for s in nlp(ex["context_en"]).sentences
                    ]
                )
                decoder_input = model.encode(ctx_tgt, as_targets=True).to(model.device)
                generate_kwargs["decoder_input_ids"] = decoder_input.input_ids
                if has_lang_tag:
                    lang_id_tensor = torch.tensor([model.tokenizer.lang_code_to_id["fr_XX"]]).to(model.device)
                    # Prepend the ID tensor to the original tensor along the first dimension (rows)
                    generate_kwargs["decoder_input_ids"] = torch.cat(
                        (lang_id_tensor.unsqueeze(0), generate_kwargs["decoder_input_ids"]), dim=1
                    )
            encoded_sources = model.encode(contrast_sources, as_targets=False).to(model.device)
            generation_out = model.model.generate(
                input_ids=encoded_sources.input_ids,
                attention_mask=encoded_sources.attention_mask,
                return_dict_in_generate=True,
                **generate_kwargs,
            )
            encoded_sources = encoded_sources.to("cpu")
            if not has_context:
                decoder_input = decoder_input.to("cpu")
            ctx_gen = (
                model.tokenizer.batch_decode(
                    generation_out.sequences, skip_special_tokens=False if has_target_context else True
                )[0]
                .replace("<pad>", "")
                .replace("</s>", "")
            )
            del generation_out
            torch.cuda.empty_cache()
            if has_context and has_target_context:
                ctx_tgt = ctx_gen.split("<brk>")[0].strip() + "<brk>"
            start_pos = len(ctx_tgt) if ctx_tgt else 0
            ctx_gen = ctx_gen[start_pos:]
        tgt = ex["fr"] if force_gen else ctx_gen
        ctx_tgt = ex["context_fr"] if force_gen else ctx_tgt
        examples.append(
            {
                "src_en": ex["en"],
                "tgt_fr": tgt,
                "src_en_ctx": contrast_sources,
                "tgt_fr_ctx": ctx_tgt,
                "src_en_with_tags": ex["en_with_tags"],
                "orig_fr": ex["fr"],
                "orig_fr_with_tags": ex["fr_with_tags"],
            }
        )
        if idx < 3:
            print(f"FULL EXAMPLE: {ex}")
            print(f"SRC: {ex['en']}")
            print(f"TGT: {tgt}")
            print(f"SRC CTX: {contrast_sources}")
            print(f"TGT CTX: {ctx_tgt}")
    return examples
