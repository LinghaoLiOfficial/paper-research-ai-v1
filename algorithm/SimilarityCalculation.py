class SimilarityCalculation:

    @classmethod
    def weighted_reverse_simrank(cls, out_edges, C=0.8, max_iter=100, threshold=1e-4):
        """

        :param out_edges:
        :param C: 衰减系数
        :param max_iter: 最大迭代次数
        :param threshold: 阈值
        :return: 相似度矩阵
        """

        # 获取所有节点列表
        nodes = set(out_edges.keys())
        for node in out_edges.values():
            nodes.update(node.keys())
        nodes = sorted(nodes)

        # 初始化相似度矩阵：自己与自己完全相似，其他为0
        sim = {a: {b: 1.0 if a == b else 0.0 for b in nodes} for a in nodes}

        # 迭代更新相似度
        for _ in range(max_iter):
            # 新一轮的相似度
            new_sim = {a: {b: 0.0 for b in nodes} for a in nodes}
            # 记录当前迭代的最大变化量（用于判断收敛）
            max_diff = 0.0

            # 遍历所有节点对(a, b)
            for a in nodes:
                for b in nodes:
                    if a == b:
                        # 自己与自己的相似度始终为1
                        new_sim[a][b] = 1.0
                        continue

                    # 获取a和b的出邻居及其权重
                    out_a = out_edges.get(a, {})
                    out_b = out_edges.get(b, {})

                    # 如果a或b没有出邻居，则相似度为0
                    if not out_a or not out_b:
                        new_sim[a][b] = 0.0
                    else:
                        # 计算a和b的总入边权重(用于归一化)
                        total_weight_a = sum(out_a.values())
                        total_weight_b = sum(out_b.values())

                        # 核心计算：遍历a和b的所有出邻居对(i, j)
                        total = 0.0
                        for i in out_a:
                            for j in out_b:
                                # 加权求和：权重归一化后相乘，再乘上一轮迭代的sim[i][j]
                                total += (out_a[i] / total_weight_a) * \
                                         (out_b[j] / total_weight_b) * \
                                         sim[i][j]
                        # 乘以衰减因子C得到最终的相似度
                        new_sim[a][b] = C * total

                    # 更新最大变化量(用于判断是否提前终止迭代)
                    max_diff = max(max_diff, abs(new_sim[a][b] - sim[a][b]))

            # 如果变化量小于阈值，提前终止迭代
            if max_diff < threshold:
                break
            # 将new_sim的值更新到sim中，进入下一轮迭代
            sim = {a: {b: new_sim[a][b] for b in nodes} for a in nodes}

        # 返回最终的相似度矩阵
        return sim
