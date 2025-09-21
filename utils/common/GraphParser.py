from collections import Counter

from config.neo4j.NodeEdgeColorStr import NodeEdgeColorStr
from config.neo4j.NodeNameStr import NodeNameStr
from config.neo4j.NodeColorStr import NodeColorStr
from mapper.KnowledgeBaseMapper import KnowledgeBaseMapper
from utils.common.RandomStrGenerator import RandomStrGenerator


class GraphParser:

    @classmethod
    def science_recursive_get_tree(cls, text_id, node_map):
        node_data = node_map[text_id]
        children = []
        for child_id in [x for x in node_data['children_ids'] if x != node_data['id']]:
            if child_id in node_map.keys():
                children_node = cls.science_recursive_get_tree(child_id, node_map)
                children.append(children_node)

        if "label" in node_data.keys():
            output_data = {
                "id": node_data['id'],
                "label": f"{node_data['label']}    *{node_data['zh_label']}",
                "type": node_data['type'],
                "weight": node_data['weight'],
                "theme_explaining": node_data['theme_explaining'],
                "children": children
            }
        else:
            output_data = {
                "id": node_data['id'],
                "label": "",
                "type": node_data['type'],
                "weight": node_data['weight'],
                "theme_explaining": node_data['theme_explaining'],
                "children": children
            }

        return output_data

    @classmethod
    def recursive_get_tree(cls, text_id, node_map, image_base_path):
        node_data = node_map[text_id]
        children = []
        for child_id in [x for x in node_data['children_ids'] if x != node_data['id']]:
            if child_id in node_map.keys():
                children.append(cls.recursive_get_tree(child_id, node_map, image_base_path))

        if "label" in node_data.keys():
            output_data = {
                "id": node_data['id'],
                "label": node_data['label'],
                "children": children
            }
        elif "image_name" in node_data.keys():
            output_data = {
                "id": node_data['id'],
                "imageName": node_data['image_name'],
                "imageWidth": node_data['image_width'],
                "imageHeight": node_data['image_height'],
                "imageUrl": f"/file/{image_base_path.lstrip('./')}/{node_data['image_name']}",
                "children": children
            }
        else:
            output_data = {
                "id": node_data['id'],
                "label": "",
                "children": children
            }

        return output_data

    @classmethod
    def science_init_recursive_get_tree(cls, root_list, paper_list):

        node_list = paper_list + root_list

        if len(node_list) == 1:
            graph_data = {
                "id": root_list[0]['id'],
                "label": f"{root_list[0]['label']}    *{root_list[0]['zh_label']}",
                "type": root_list[0]['type'],
                "weight": root_list[0]['weight'],
                "theme_explaining": root_list[0]['theme_explaining'],
                # 删除children中包含的自身id
                "children": [x for x in root_list[0]['children_ids'] if x != root_list[0]['id']]
            }

            return graph_data

        node_map = {row['id']: row for row in node_list}
        root_id = root_list[0]['id']
        graph_data = cls.science_recursive_get_tree(root_id, node_map)

        return graph_data

    @classmethod
    def init_recursive_get_tree(cls, root_list, text_list, image_list, image_base_path):

        node_list = text_list + root_list + image_list

        if len(node_list) == 1:
            graph_data = {
                "id": root_list[0]['id'],
                "label": root_list[0]['label'],
                # 删除children中包含的自身id
                "children": [x for x in root_list[0]['children_ids'] if x != root_list[0]['id']]
            }

            return graph_data

        node_map = {row['id']: row for row in node_list}
        root_id = root_list[0]['id']
        graph_data = cls.recursive_get_tree(root_id, node_map, image_base_path)

        return graph_data

    @classmethod
    def recursive_create_or_update_tree(cls, parent_node):
        for child_data in parent_node['children']:
            if "imageName" in child_data.keys():
                KnowledgeBaseMapper.merge_image_node({
                    "parent_id": parent_node["id"],
                    "child_image_name": child_data['imageName'],
                    "child_image_id": child_data["id"],
                    "child_image_width": child_data['imageWidth'],
                    "child_image_height": child_data['imageHeight']
                })
            else:
                KnowledgeBaseMapper.merge_text_node({
                    "parent_id": parent_node["id"],
                    "child_text_name": child_data['label'],
                    "child_text_id": child_data["id"]
                })

            if 'children' in child_data:
                cls.recursive_create_or_update_tree(child_data)

    @classmethod
    def get_node_properties_labels_id(cls, in_list):
        out_list = []

        for x in in_list:

            name = ""
            for k, v in dict(x[0]).items():
                if "name" in k:
                    name = v
                    break

            out_list.append({
                "id": x[2],
                "name": name
            })

        return out_list

    @classmethod
    def get_properties(cls, in_list: list, name_mapping: dict = {}):
        out_list = []
        for x in in_list:
            current_dict = {}
            for k, v in dict(x['properties']).items():
                if name_mapping != {} and k in name_mapping.keys():
                    current_dict[name_mapping[k]] = v
                else:
                    current_dict[k] = v
            out_list.append(current_dict)

        return out_list

    @classmethod
    def get_color(cls):
        node_color = NodeColorStr.mapping()
        node_edge_color = NodeEdgeColorStr.mapping()

        return node_color, node_edge_color

    @classmethod
    def count_label_type(cls, node_list):
        total_list = [x["label"] for x in node_list]

        count_dict = dict(Counter(total_list))
        count_dict["*"] = sum([x for x in count_dict.values()])

        return count_dict

    @classmethod
    def convert_graph_into_node_and_relation(cls, n, n_list: list):

        # 转换节点和边

        node_list = []
        relation_list = []

        for layer in n_list:
            layer_list = []
            for x in layer:
                current_dict = dict(x[0])
                current_dict["label"] = x[1][0]
                current_dict["name"] = current_dict[NodeNameStr.mapping()[x[1][0]]]
                current_dict["id"] = x[3]
                current_dict["momentum"] = 0
                layer_list.append(current_dict)

                current_dict1 = {
                    "source": x[2],
                    "target": x[3],
                    "momentum": 0
                }
                relation_list.append(current_dict1)

            node_list.append(layer_list)

        layer_list = []
        current_dict = dict()
        current_dict["label"] = n[0][1][0]
        current_dict["id"] = n[0][2]
        current_dict["momentum"] = 0

        current_dict.update(dict(n[0][0]))
        current_dict["name"] = current_dict[NodeNameStr.mapping()[n[0][1][0]]]

        layer_list.append(current_dict)
        node_list = [layer_list] + node_list

        # 计算各条边的势能

        for i in range(len(node_list) - 1, 0, -1):
            for node in node_list[i]:
                new_momentum = node["momentum"] + 1

                found_relation_index = [(index, relation_list[index]["source"]) for index in range(len(relation_list))
                                        if relation_list[index]["target"] == node["id"]]
                source_node_id_list = []
                for pair in found_relation_index:
                    relation_list[pair[0]]["momentum"] = new_momentum
                    source_node_id_list.append(pair[1])

                for source_node_id in source_node_id_list:
                    for j in range(len(node_list[i - 1])):
                        if node_list[i - 1][j]["id"] == source_node_id:
                            node_list[i - 1][j]["momentum"] = max(new_momentum, node_list[i - 1][j]["momentum"])

        # 拆分节点层级

        combined_node_list = []
        for layer in node_list:
            combined_node_list.extend(layer)

        return combined_node_list, relation_list
