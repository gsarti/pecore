from strenum import StrEnum


class ModelTypeEnum(StrEnum):
    MBART = "mbart50-1toM"
    MARIAN_BIG = "marian-big"
    MARIAN_SMALL = "marian-small"


class DatasetEnum(StrEnum):
    FLORES = "flores"
    IWSLT = "iwslt17"
    SCAT = "scat"


class MetricEnum(StrEnum):
    BLEU = "bleu"
    COMET = "comet"
    ACCURACY = "accuracy"


class AttributeFnEnum(StrEnum):
    BASE = "base"
    TOP_P = "top_p"
    LOGIT_LENS = "logit_lens"
    INPUT_CONTRIBUTIONS = "input_contributions"
