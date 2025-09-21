from collections import Counter

import pandas as pd


class DataViewer:

    @classmethod
    def view_label_num(cls, iteration) -> Counter:

        counter = Counter(iteration)

        return counter
