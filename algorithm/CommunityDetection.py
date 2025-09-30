from infomap import Infomap
import random


class CommunityDetection:

    @classmethod
    def infomap(cls, nodes, edges):

        """
            指标:CodeLength: 编码长度越小越好，通常为2-10
            指标:Modularity: 模块度越高社区结构越明显，0.3-0.7为佳
            指标:Community Size SD:社区规模标准差越小越均衡

            结合领域知识人工抽查社区内论文的主题一致性进行验证
        """

        # TODO: 优化：移除低权重边（示例：保留权重>2的边）

        # 创建映射：节点ID -> 索引（Infomap需要整数节点）
        node_to_id = {node: idx for idx, node in enumerate(nodes)}

        # 初始化Infomap对象
        im = Infomap(
            flow_model="directed",
            seed=42
        )

        # 添加节点和边
        for node in nodes:
            im.add_node(node_to_id[node])

        for source, target, weight in edges:
            im.add_link(node_to_id[source], node_to_id[target], weight)

        # 运行社区检测
        im.run()

        # 提取社区结构
        communities = {}
        for node, idx in node_to_id.items():
            community_id = im.get_modules().get(idx)
            if community_id not in communities:
                communities[community_id] = []
            communities[community_id].append(node)

        # 模块度评估
        print(f"模块度: {im.codelength:.4f} bits")

        # 社区大小降序排序
        community_result = list(communities.values())
        sorted_community_result = sorted(community_result, key=lambda v: len(v), reverse=True)

        return sorted_community_result
