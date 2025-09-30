from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np


class ClusterClassification:
    @classmethod
    def hierarchical_clustering(cls, S):
        # 输入相似度矩阵S（需转换为距离矩阵：距离 = 1 - 相似度）
        Z = linkage(S, method='average')
        t = 2
        labels = fcluster(Z, t=t, criterion='distance')

        cluster_num = len(np.unique(labels))
        print(f"聚类后簇的数量(t:{t}): {cluster_num}")

        return labels.tolist(), cluster_num
