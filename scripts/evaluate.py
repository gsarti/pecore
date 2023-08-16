import argparse
import logging
from pathlib import Path

from comet import download_model, load_from_checkpoint
from pecore.alignment_utils import get_aligned_gender_annotations
from pecore.data_utils import get_src_ref_sentences, load_mt_dataset
from pecore.enums import DatasetEnum, MetricEnum, ModelTypeEnum
from pecore.model_utils import get_model_id
from sacrebleu.metrics import BLEU
from tqdm import tqdm

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
        "--model_id",
        type=str,
        default=None,
        help="Identifier to use for the model outputs. If not specified, will build from model type",
    )
    parser.add_argument(
        "--use_target_context",
        action="store_true",
        help="Whether to use target context for translation",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        choices=[m.value for m in MetricEnum],
        default=[MetricEnum.BLEU],
        help="Metrics to use for evaluation",
    )
    args = parser.parse_args()
    return args


def evaluate():
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
    data = load_mt_dataset(args.dataset, args.src_lang, args.tgt_lang)
    src, refs = get_src_ref_sentences(
        dataset_name=args.dataset,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        dataset=data,
    )
    with open(Path(args.output_dir) / f"{args.dataset}-{model_id}.txt") as f:
        sys = f.readlines()
    if args.use_target_context:
        # Remove target context for evaluation
        sys = [s.split("<brk>")[1].strip() if "<brk>" in s else s for s in sys]
    for metric in args.metrics:
        if metric == MetricEnum.COMET:
            model_path = download_model("Unbabel/wmt22-comet-da")
            model = load_from_checkpoint(model_path)
            comet_out = model.predict(
                [{"src": s, "mt": m, "ref": r} for s, m, r in zip(src, sys, refs)], batch_size=8, gpus=1
            )
            print(args.dataset, model_id, "COMET", comet_out.system_score)
        elif metric == MetricEnum.BLEU:
            bleu = BLEU()
            print(args.dataset, model_id, bleu.corpus_score(sys, [refs]))
        if metric == MetricEnum.ACCURACY:
            if args.dataset != DatasetEnum.SCAT:
                raise ValueError("Only SCAT supports accuracy metric")
            tot_keywords, tot_correct = 0, 0
            ref_contrast = data[f"contrast_{args.tgt_lang[:2]}"]
            for curr_ref, curr_ref_contrast, curr_mt in tqdm(
                zip(refs, ref_contrast, sys), desc="Aligned accuracy", total=len(refs)
            ):
                matches = get_aligned_gender_annotations(curr_ref, curr_ref_contrast, curr_mt)
                tot_keywords += len(matches)
                tot_correct += len([x for x in matches if x == 1])
            print(args.dataset, model_id, "align_match_acc", round(tot_correct / tot_keywords, 4))
