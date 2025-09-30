import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import normalized_mutual_info_score


def calculate_mutual_information(array1: np.ndarray, array2: np.ndarray) -> float:

    # 计算两个array相应列之间的互信息

    mi_scores = []
    for i in range(array1.shape[1]):

        mi_score = normalized_mutual_info_score(array1[:, i].tolist(), array2[:, i].tolist())

        mi_scores.append(mi_score)

    max_mi_score = max(mi_scores)

    return max_mi_score


def calculate_cosine_similarity(array1: np.ndarray, array2: np.ndarray) -> float:

    dot_product = np.sum(array1 * array2, axis=0)

    norm_A = np.linalg.norm(array1, axis=0)
    norm_B = np.linalg.norm(array2, axis=0)

    norm = norm_A * norm_B

    cosine_sim = [dot_product[i] / norm[i] if norm[i] != 0 else 0 for i in range(len(dot_product))]

    avg_cosine_sim = sum(cosine_sim) / len(cosine_sim)

    return avg_cosine_sim


class GraphDataProcess:

    k_bins_discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')

    @classmethod
    def get_time_context_adj(cls, time_array: np.ndarray, time_context_adj_k: int):

        turbine_list = np.split(time_array, time_array.shape[1], axis=1)

        turbine_list = [x.reshape(time_array.shape[0], time_array.shape[2]) for x in turbine_list]

        length = len(turbine_list)

        temporal_df = pd.DataFrame(
            data=np.zeros((length, length)),
            index=[x for x in range(length)],
            columns=[x for x in range(length)]
        )
        for i in range(length):

            for j in range(length):

                score = calculate_cosine_similarity(turbine_list[i], turbine_list[j])

                temporal_df.iloc[i, j] = score

        for i in range(len(temporal_df)):
            temporal_df.iloc[i, i] = float("inf")

        adj = cls.generate_adj(
            df=temporal_df,
            k=time_context_adj_k
        )

        return adj

    @classmethod
    def get_distance_adj(cls, turbine_to_delete: list, distance_adj_k: int):

        spatial_df = pd.read_csv("./data/su_you/spatial/distance_matrix.csv", index_col=0)

        valid_df_index_list = [x for x in range(50) if (x + 1) not in turbine_to_delete]
        spatial_df = spatial_df.iloc[valid_df_index_list, valid_df_index_list].reset_index().drop("index", axis=1)
        spatial_df.columns = [x for x in spatial_df.index]

        for i in range(len(spatial_df)):
            spatial_df.iloc[i, i] = float("inf")

        adj = cls.generate_adj(
            df=spatial_df,
            k=distance_adj_k
        )

        return adj

    @classmethod
    def generate_adj(cls, df: pd.DataFrame, k: int):

        distance_dict = {}
        for i in range(len(df)):
            distance_dict[df.index.values[i]] = df.values[i].tolist()

        index_sorted_distance_dict = {}
        for number, distances in distance_dict.items():
            sorted_distances = sorted(distances)
            mapped_distance_list = [distances.index(x) for x in sorted_distances][:k]
            index_sorted_distance_dict[number] = mapped_distance_list

        adj = np.zeros((len(df.columns), len(df.index)))

        for number, distances in index_sorted_distance_dict.items():
            for dist in distances:
                adj[number, dist] = 1

        return adj



