from collections import Counter, OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, load_dataset

from .enums import DatasetEnum

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
