from config.name.BaseName import BaseName


class CrawlTaskParamDefault(BaseName):
    SEARCH_WORDS = "wind turbine failure predict"
    SUBJECT_FILTER_ECONOMICS = 0
    SUBJECT_FILTER_BUSINESS = 0
    SUBJECT_FILTER_ENGINEERING = 1
    SUBJECT_FILTER_APPLIED_SCIENCE = 1
    MAX_NUM = 30