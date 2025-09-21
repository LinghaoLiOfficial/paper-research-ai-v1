from fuzzywuzzy import fuzz
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class DistanceCalculation:

    @classmethod
    def str_calculate_levenshtein_distance(cls, str_1: str, str_2: str):
        similarity = fuzz.ratio(str_1, str_2)

        return similarity

    @classmethod
    def calculate_cosine_similarity(cls, vector_1, vector_2):
        if isinstance(vector_1, list):
            vector_1 = np.array(vector_1)
        if isinstance(vector_2, list):
            vector_2 = np.array(vector_2)

        cos_sim = np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))

        return cos_sim


