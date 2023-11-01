import argparse
import logging
from pathlib import Path
from pprint import pprint

import pandas as pd
import stanza
import torch
from pecore.data_utils import load_mt_dataset
from pecore.enums import DatasetEnum, ModelTypeEnum
from pecore.model_utils import get_lang_from_model_type, has_lang_tag
from tqdm import tqdm

import inseq

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
        default="outputs/processed_examples",
        help="Root folder to save processed examples outputs",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[d.value for d in DatasetEnum],
        required=True,
        help="Dataset to use to generate examples.",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="Dataset config to use.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Full HF Hub id fo the model to use to generate examples.",
    )
    parser.add_argument(
        "--src_lang",
        type=str,
        default="eng",
        help="Source language to use for evaluation.",
    )
    parser.add_argument(
        "--tgt_lang",
        type=str,
        default="fra",
        help="Target language to use for evaluation.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="",
        help="Identifier to use for the model outputs. If not specified, will build from model type",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=[t.value for t in ModelTypeEnum],
        default=None,
        help="Model type to use for translation. If not specified, will use the model name.",
    )
    parser.add_argument(
        "--context_column",
        type=str,
        default="context_{lang}",
        help="Column name identifying context sentences in the dataset.",
    )
    parser.add_argument(
        "--current_column",
        type=str,
        default="{lang}",
        help="Column name identifying current sentences in the dataset.",
    )
    parser.add_argument(
        "--current_tagged_column",
        type=str,
        default="{lang}_with_tags",
        help="Column name identifying tagged currentsentences in the dataset.",
    )
    parser.add_argument(
        "--context_tagged_column",
        type=str,
        default="context_{lang}_with_tags",
        help="Column name identifying tagged context sentences in the dataset.",
    )
    parser.add_argument(
        "--contrast_column",
        type=str,
        default="contrast_{lang}",
        help="Column name identifying contrast sentences in the dataset.",
    )
    parser.add_argument(
        "--use_gold_target",
        action="store_true",
        help="Whether to use gold targets instead of generating.",
    )
    parser.add_argument(
        "--has_context",
        action="store_true",
        help="Whether the model uses context (<brk> tags to separate context and current sentences)",
    )
    parser.add_argument(
        "--has_target_context",
        action="store_true",
        help=(
            "Whether the model generate context alongside current targets (<brk> tags to separate context and current"
            " sentences)"
        ),
    )
    parser.add_argument(
        "--has_contrast",
        action="store_true",
        help="Whether the dataset has contrast sentences.",
    )
    parser.add_argument(
        "--special_tokens",
        type=str,
        nargs="+",
        default=["<pad>", "</s>"],
    )
    args = parser.parse_args()
    return args


def format_examples():
    args = parse_args()
    data = load_mt_dataset(args.dataset, args.src_lang, args.tgt_lang, dataset_config=args.dataset_config)
    model = inseq.load_model(args.model_name, "dummy")
    nlp = stanza.Pipeline(lang=args.src_lang[:2], processors="tokenize", download_method=None)
    src_context_column = args.context_column
    tgt_context_column = args.context_column
    src_current_column = args.current_column
    tgt_current_column = args.current_column
    src_context_tagged_column = args.context_tagged_column
    tgt_context_tagged_column = args.context_tagged_column
    src_current_tagged_column = args.current_tagged_column
    tgt_current_tagged_column = args.current_tagged_column
    contrast_column = args.contrast_column
    if "{lang}" in args.context_column:
        src_context_column = src_context_column.format(lang=args.src_lang[:2])
        tgt_context_column = tgt_context_column.format(lang=args.tgt_lang[:2])
    else:
        src_context_column = "src_" + src_context_column
        tgt_context_column = "tgt_" + tgt_context_column
    if "{lang}" in args.current_column:
        src_current_column = src_current_column.format(lang=args.src_lang[:2])
        tgt_current_column = tgt_current_column.format(lang=args.tgt_lang[:2])
    else:
        src_current_column = "src_" + src_current_column
        tgt_current_column = "tgt_" + tgt_current_column
    if "{lang}" in args.context_tagged_column:
        src_context_tagged_column = src_context_tagged_column.format(lang=args.src_lang[:2])
        tgt_context_tagged_column = tgt_context_tagged_column.format(lang=args.tgt_lang[:2])
    else:
        src_context_tagged_column = "src_" + src_context_tagged_column
        tgt_context_tagged_column = "tgt_" + tgt_context_tagged_column
    if "{lang}" in args.current_tagged_column:
        src_current_tagged_column = src_current_tagged_column.format(lang=args.src_lang[:2])
        tgt_current_tagged_column = tgt_current_tagged_column.format(lang=args.tgt_lang[:2])
    else:
        src_current_tagged_column = "src_" + src_current_tagged_column
        tgt_current_tagged_column = "tgt_" + tgt_current_tagged_column
    if "{lang}" in args.contrast_column:
        contrast_column = contrast_column.format(lang=args.tgt_lang[:2])
    examples = []
    for _idx, ex in tqdm(enumerate(data), total=len(data)):
        generate_kwargs = {}
        if has_lang_tag(model):
            model.tokenizer.src_lang = get_lang_from_model_type(args.model_type, args.src_lang)
            model.tokenizer.tgt_lang = get_lang_from_model_type(args.model_type, args.tgt_lang)
            generate_kwargs["forced_bos_token_id"] = model.tokenizer.lang_code_to_id[model.tokenizer.tgt_lang]
            args.special_tokens.append(model.tokenizer.tgt_lang)
        if args.has_context:
            source = ex[src_context_column].strip() + "<brk> " + ex[src_current_column]
            if args.has_target_context:
                target = ex[tgt_context_column].strip() + "<brk> " + ex[tgt_current_column]
            else:
                target = ex[tgt_current_column]
        else:
            source = ex[src_context_column].strip() + " " + ex[src_current_column]
            target = ex[tgt_context_column].strip() + " " + ex[tgt_current_column]
        curr_example = {
            "source_full": source,
            "source_current": ex[src_current_column],
            "source_context": ex[src_context_column],
            "source_current_tagged": ex[src_current_tagged_column],
            "source_context_tagged": ex[src_context_tagged_column],
            "gold_target_full": target,
            "gold_target_current": ex[tgt_current_column],
            "gold_target_context": ex[tgt_context_column],
            "gold_target_current_tagged": ex[tgt_current_tagged_column],
            "gold_target_context_tagged": ex[tgt_context_tagged_column],
        }
        if args.has_contrast:
            curr_example["gold_target_current_contrast"] = ex[contrast_column]
        if args.use_gold_target:
            examples.append(curr_example)
        else:
            generated_target_context = None
            if not args.has_context:
                # Create a generated target context by translating the source context
                # sentence-by-sentence and using it as target prefix for generating the current target.
                generated_target_context = " ".join(
                    [
                        model.generate(sent.text, max_new_tokens=128, **generate_kwargs)[0]
                        for sent in nlp(ex[src_context_column]).sentences
                    ]
                )
                # decoder_input = model.encode(
                #    generated_target_context, as_targets=True, add_bos_token=not has_lang_tag(model)
                # ).to(model.device)
                # generate_kwargs["decoder_input_ids"] = decoder_input.input_ids
                generated_target_current = model.generate(ex[src_current_column], **generate_kwargs)[0]
                torch.cuda.empty_cache()
                generated_target = generated_target_context + " " + generated_target_current
            else:
                generated_target = model.generate(
                    source, skip_special_tokens=not args.has_target_context, **generate_kwargs
                )[0].strip()
                generated_target_current_noctx = model.generate(
                    ex[src_current_column], skip_special_tokens=True, **generate_kwargs
                )[0].strip()
                for st in args.special_tokens:
                    generated_target = generated_target.replace(st, "").strip()
                torch.cuda.empty_cache()
                generated_target_split = generated_target.split("<brk>")
                if args.has_context and args.has_target_context:
                    if len(generated_target_split) > 0:
                        generated_target_context = generated_target_split[0].strip()
                if generated_target_context is not None:
                    if len(generated_target_split) > 1:
                        generated_target_current = generated_target_split[1].strip()
                    else:
                        generated_target_current = None
                else:
                    generated_target_current = generated_target
            curr_example["generated_target_full"] = generated_target
            curr_example["generated_target_current"] = generated_target_current
            curr_example["generated_target_context"] = generated_target_context
            curr_example["generated_target_current_noctx"] = generated_target_current_noctx
            examples.append(curr_example)
        if _idx < 3:
            pprint(curr_example)
    df = pd.DataFrame(examples)
    save_path = (
        Path(args.output_dir)
        / f"{args.dataset}{'-' + args.dataset_config if args.dataset_config else ''}-{args.model_id}.tsv"
    )
    df.to_csv(save_path, index=False, sep="\t")
    logger.info(f"{len(df)} examples saved to {save_path}")


if __name__ == "__main__":
    format_examples()
