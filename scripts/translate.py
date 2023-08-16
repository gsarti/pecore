import argparse
import logging
from pathlib import Path

import torch
from pecore.data_utils import get_preprocess_dataset_fn, load_mt_dataset
from pecore.enums import DatasetEnum, ModelTypeEnum
from pecore.model_utils import (
    encode_examples,
    get_lang_from_model_type,
    get_model_attribute,
    get_model_id,
    get_model_name,
    has_lang_tag,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s  %(message)s",
    datefmt="%d-%m-%y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/translations",
        help="Root folder containing translation outputs",
    )
    parser.add_argument(
        "--context_word_dropout",
        type=int,
        default=0,
        help="Context word dropout percentage for the selected model (e.g. 2 = 20%)",
    )
    parser.add_argument(
        "--context_size",
        type=int,
        default=0,
        help="Number of previous sentences to use as context",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=[t.value for t in ModelTypeEnum],
        default=None,
        help="Model type to use for translation. If not specified, will use the model name.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[d.value for d in DatasetEnum],
        required=True,
        help="Dataset to use for translation.",
    )
    parser.add_argument(
        "--src_lang",
        type=str,
        default="eng",
        help="Source language to use for translation.",
    )
    parser.add_argument(
        "--tgt_lang",
        type=str,
        default="fra",
        help="Target language to use for translation.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Full HF Hub id fo the model to use for translation. If not specified, will use the model type.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Identifier to use for the model outputs. If not specified, will build from model type",
    )
    args = parser.parse_args()
    return args


def translate():
    args = parse_args()
    if args.model_id is None:
        model_id = get_model_id(
            dataset=args.dataset,
            model_type=args.model_type,
            context_size=args.context_size,
            context_word_dropout=args.context_word_dropout,
        )
    else:
        model_id = args.model_type
    if args.model_name is None:
        model_name = get_model_name(
            model_id=model_id,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
        )
    else:
        model_name = args.model_name
    model_id += "-noctx" if args.context_size == 0 else ""
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    has_lang_prefix = has_lang_tag(model)
    tok_kwargs = {}
    if has_lang_prefix:
        tok_kwargs["src_lang"] = get_lang_from_model_type(args.model_type, args.src_lang)
        tok_kwargs["tgt_lang"] = get_lang_from_model_type(args.model_type, args.tgt_lang)
    tok = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
    preproc_fn = get_preprocess_dataset_fn(
        context_size=args.context_size,
        dataset=args.dataset,
        src_lang=args.src_lang,
    )
    data = load_mt_dataset(args.dataset, args.src_lang, args.tgt_lang)
    data_preproc = data.map(preproc_fn, batched=True, batch_size=2000, remove_columns=data.column_names)
    data_tokenized = data_preproc.map(lambda x: encode_examples(x, tok), batched=True, remove_columns=["sentence"])
    data_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    dataloader = DataLoader(data_tokenized, batch_size=get_model_attribute(args.model_type, "batch_size"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    logger.info("Translating...")
    out_folder = "ctx" if args.context_size > 0 else "noctx"
    with open(Path(args.output_dir) / out_folder / f"{args.dataset}-{args.model_id}.txt", "a") as f:
        gen_kwargs = {}
        if has_lang_prefix:
            gen_kwargs = {"forced_bos_token_id": tok.lang_code_to_id[tok_kwargs["tgt_lang"]]}
        for batch in tqdm(dataloader):
            device_batch = {k: v.to(device) for k, v in batch.items()}
            out = model.generate(**device_batch, **gen_kwargs)
            if args.context_size > 0:
                translations = tok.batch_decode(out.to("cpu"), skip_special_tokens=False)
                translations = [
                    t.replace("<pad>", "").replace("</s>", "").replace(tok_kwargs["tgt_lang"], "").strip()
                    for t in translations
                ]
            else:
                translations = tok.batch_decode(out.to("cpu"), skip_special_tokens=True)
            for trans in translations:
                f.write(trans + "\n")


if __name__ == "__main__":
    translate()
