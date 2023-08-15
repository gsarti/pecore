import os
from collections import Counter, OrderedDict

import datasets
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

flores = datasets.load_dataset("gsarti/flores_101", "all", split="devtest")
iwslt = datasets.load_dataset("gsarti/iwslt2017_context", "iwslt2017-en-fr", split="test")
scat = datasets.load_dataset("inseq/scat", split="test", verification_mode="no_checks")

DATASETS = {"scat": scat, "flores": flores, "iwslt17": iwslt}
BASE_PATH = "translations/translations"


# Used in dataset preprocessing to sample the number of context sentences to include
class OrderedCounter(Counter, OrderedDict):
    "Counter that remembers the order elements are first encountered"

    def __repr__(self):
        return f"{self.__class__.__name__}({OrderedDict(self)!r})"

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


def get_preprocess_dataset(ctx_size, dataset="flores", src_lang="eng"):
    def preprocess_dataset_seq(examples):
        if dataset == "flores":
            inputs = examples[f"sentence_{src_lang[:3]}"]
            n_previous = [i for _, v in OrderedCounter(examples["URL"]).items() for i in range(v)]
        elif dataset == "iwslt17":
            inputs = [ex[src_lang[:2]] for ex in examples["translation"]]
            n_previous = [i for _, v in OrderedCounter(examples["doc_id"]).items() for i in range(v)]
        else:
            raise ValueError(f"Not available: {dataset}")
        n_contexts = [ctx_size for _ in range(len(inputs))]
        n_contexts = [min(n_contexts[idx], n_previous[idx]) for idx in range(len(n_contexts))]
        context_inputs = []
        for idx in range(len(inputs)):
            if n_contexts[idx] > 0:
                ctx = " ".join(inputs[idx - n_contexts[idx] : idx])
                context_inputs.append(f"{ctx}<brk> {inputs[idx]}")
            else:
                context_inputs.append(inputs[idx])
        return {"sentence": context_inputs}

    def preprocess_dataset_merged(examples):
        if dataset == "scat":
            inputs = examples[src_lang[:2]]
            contexts = examples[f"context_{src_lang[:2]}"]
        context_inputs = []
        for idx in range(len(inputs)):
            if ctx_size > 0:
                context_inputs.append(f"{contexts[idx]}<brk> {inputs[idx]}")
            else:
                context_inputs.append(inputs[idx])
        return {"sentence": context_inputs}

    if dataset in ["flores", "iwslt17"]:
        return preprocess_dataset_seq
    elif dataset in ["scat"]:
        return preprocess_dataset_merged


def encode(examples, tokenizer):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length")


def translate(
    cwd,
    ctx,
    model_type,
    dataset="flores",
    src_lang="eng",
    use_context=True,
    has_lang_tag=False,
    model_name: str = None,
):
    if model_name is None:
        model_id = f"{model_type}-ctx{ctx}-cwd{cwd}"
        model_name = f"context-mt/iwslt17-{model_id}-en-fr"
    else:
        model_id = model_type
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if not has_lang_tag:
        tok = AutoTokenizer.from_pretrained(model_name)
    else:
        tok = AutoTokenizer.from_pretrained(model_name, src_lang="en_XX", tgt_lang="fr_XX")
    preproc_fn = get_preprocess_dataset(ctx if use_context else 0, dataset=dataset, src_lang=src_lang)
    data_preproc = DATASETS[dataset].map(
        preproc_fn, batched=True, batch_size=2000, remove_columns=DATASETS[dataset].column_names
    )
    data_tokenized = data_preproc.map(lambda x: encode(x, tok), batched=True, remove_columns=["sentence"])
    data_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    dataloader = torch.utils.data.DataLoader(
        data_tokenized, batch_size=8 if "marian-small" in model_type else 4 if "marian-big" in model_type else 1
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    print("Translating...")
    with open(os.path.join(BASE_PATH, f"{dataset}-{model_id}{'-noctx' if not use_context else ''}.txt"), "a") as f:
        for i, batch in enumerate(tqdm(dataloader)):
            if not has_lang_tag:
                out = model.generate(**{k: v.to(device) for k, v in batch.items()})
            else:
                out = model.generate(**batch, forced_bos_token_id=tok.lang_code_to_id["fr_XX"])
            if use_context:
                translations = tok.batch_decode(out.to("cpu"), skip_special_tokens=False)
                translations = [
                    t.replace("<pad>", "").replace("</s>", "").replace("fr_XX", "").strip() for t in translations
                ]
            else:
                translations = tok.batch_decode(out.to("cpu"), skip_special_tokens=True)
            if i == 0:
                print(translations[:2])
            for trans in translations:
                f.write(trans + "\n")
