from config.name.BaseName import BaseName


class AnalysisTargetValueZhName(BaseName):
    MULTI_LABELS = "多标签分类"
    ALL_BINARIZE = "所有标签二值化"
    SPECIFIC_TARGET = "仅限特定标签"

    SINGLE_VAR = "单变量回归"
    MULTI_VAR = "多变量回归"
