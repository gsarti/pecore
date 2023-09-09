from strenum import StrEnum


class ModelTypeEnum(StrEnum):
    MBART = "mbart50-1toM"
    MARIAN_BIG = "marian-big"
    MARIAN_SMALL = "marian-small"


class DatasetEnum(StrEnum):
    FLORES = "flores"
    IWSLT = "iwslt17"
    SCAT = "scat"
    DISC_EVAL_MT = "disc_eval_mt"


class MetricEnum(StrEnum):
    BLEU = "bleu"
    COMET = "comet"
    ACCURACY = "accuracy"


class AttributeFnEnum(StrEnum):
    BASE = "base"
    TOP_P = "top_p"
    LOGIT_LENS = "logit_lens"
    INPUT_CONTRIBUTIONS = "input_contributions"


class EvalModeEnum(StrEnum):
    CTI = "cti"
    CCI = "cci"


class CTIMetricsEnum(StrEnum):
    RANDOM = "random"
    PCXMI = "pcxmi"
    KL_DIVERGENCE = "kl_divergence"
    CTI_MIX = "cti_mix"


class CCIMetricsEnum(StrEnum):
    RANDOM = "random"
    GRAD_PROB_DIFF = "saliency_contrast_prob_diff"
    GRAD_KL_DIV = "saliency_kl_divergence"
    IXG_PROB_DIFF = "input_x_gradient_contrast_prob_diff"
    IXG_KL_DIV = "input_x_gradient_kl_divergence"
    ATTN_ALLMEAN = "attention_default"
    ATTN_BEST = "attention_best"


class TaggedDatasetEnum(StrEnum):
    SCAT = "scat"
    DISC_EVAL_MT = "disc_eval_mt"
