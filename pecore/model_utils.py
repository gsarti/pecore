from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import yaml
from inseq import AttributionModel
from transformers import (
    M2M100ForConditionalGeneration,
    MBartForConditionalGeneration,
    NllbMoeForConditionalGeneration,
    PreTrainedModel,
    PreTrainedTokenizer,
)


@dataclass
class ModelConfig:
    batch_size: int
    num_layers: int
    num_heads: int
    lang_map: Dict[str, str] = field(default_factory=dict)


MODEL_CONFIGS = {
    model_type: ModelConfig(**cfg)
    for model_type, cfg in yaml.safe_load(open(Path(__file__).parent / "model_config.yaml", encoding="utf8")).items()
}


def encode_examples(examples: Dict[str, Any], tokenizer: PreTrainedTokenizer):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length")


def get_model_id(
    dataset: Optional[str] = None,
    model_type: Optional[str] = None,
    context_size: Optional[int] = None,
    context_word_dropout: Optional[int] = None,
) -> str:
    return f"{dataset}-{model_type}-ctx{context_size}-cwd{context_word_dropout}"


def get_model_name(
    dataset: Optional[str] = None,
    model_type: Optional[str] = None,
    context_size: Optional[int] = None,
    context_word_dropout: Optional[int] = None,
    model_id: Optional[str] = None,
    src_lang: Optional[str] = None,
    tgt_lang: Optional[str] = None,
) -> Tuple[str, str]:
    if model_id is None:
        model_id = get_model_id(
            dataset=dataset,
            model_type=model_type,
            context_size=context_size,
            context_word_dropout=context_word_dropout,
        )
    return f"context-mt/{model_id}-{src_lang[:2]}-{tgt_lang[:2]}"


def has_lang_tag(model: Union[PreTrainedModel, AttributionModel]) -> bool:
    if isinstance(model, AttributionModel):
        hf_model = model.model
    else:
        hf_model = model
    return isinstance(
        hf_model, (M2M100ForConditionalGeneration, MBartForConditionalGeneration, NllbMoeForConditionalGeneration)
    )


def get_model_attribute(model_type: str, attr_name: str) -> Any:
    if model_type not in MODEL_CONFIGS.keys():
        raise ValueError(f"Model type {model_type} not supported.")
    if not hasattr(MODEL_CONFIGS[model_type], attr_name):
        raise ValueError(f"Attribute {attr_name} not supported for model type {model_type}.")
    return getattr(MODEL_CONFIGS[model_type], attr_name)


def get_lang_from_model_type(model_type: str, lang: str) -> str:
    lang_map = get_model_attribute(model_type, "lang_map")
    if lang not in lang_map.keys():
        raise ValueError(f"Language {lang} not supported for model type {model_type}.")
    return lang_map[lang]
