import shutil

from algorithm.CommunityDetection import CommunityDetection
from buffer.KnowledgeGraphTaskBuffer import KnowledgeGraphTaskBuffer
from entity.common.Resp import Resp
from mapper.ScienceResearchMapper import ScienceResearchMapper
from buffer.ScienceResearchProgress import ScienceResearchProgress
from utils.data_analysis.DataVisualize import DataVisualize
from utils.common.GraphParser import GraphParser
from utils.common.JWTParser import JWTParser
from utils.common.KnowledgeFindingTask import KnowledgeFindingTask
from utils.common.RandomStrGenerator import RandomStrGenerator

import os

from utils.common.TimeParser import TimeParser


class ScienceResearchService:

    PAPER_PDF_PATH = "./storage/common/science_research/pdf_paper"
    FILE_PATH = "./storage/{}/knowledge_base/{}"
    TIME_EVOLVE_IMAGE_PATH = "./storage/{}/science_research/{}/time_evolve_image"
    MERMAID_IMAGE_PATH = "./storage/{}/science_research/{}/mermaid_image"

    @classmethod
    def get_time_evolve_trend(cls, task_id, content_perspective):

        mysql_result = ScienceResearchMapper.on_task_id_get_user_id({
            "task_id": task_id
        })

        user_id = mysql_result.get_data_on_results()[0]['user_id']

        time_evolve_image_path = cls.TIME_EVOLVE_IMAGE_PATH.format(user_id, task_id)

        wordcloud_image_path_list = []
        i = 1
        for path in os.listdir(time_evolve_image_path):
            if content_perspective in path:
                wordcloud_image_path_list.append({
                    "name": f'{i}.{path.split("_")[-1].split(".")[0]} Wordcloud',
                    "url": f"{os.getenv('URL')}/file{time_evolve_image_path.lstrip('.')}/{path}"
                })
                i += 1

        return Resp.build_success(data={
            "wordcloudImagePathList": wordcloud_image_path_list
        })

    @classmethod
    def get_knowledge_graph_summary(cls, task_id, content_perspective):

        mysql_result = ScienceResearchMapper.select_graph({
            "task_id": task_id,
            "content_perspective": content_perspective
        })

        literature_review = mysql_result.get_data_on_results()[0]['paragraph']

        return Resp.build_success(data={
            "literatureReview": literature_review
        })

    @classmethod
    def task_retry_task(cls, task_id, token):

        # 解析token
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        # 每个用户单次只能执行一个任务
        mysql_result0 = ScienceResearchMapper.select_paper_search_where_user_id({
            "user_id": user_id
        })

        task_status_list = [record['task_status'] for record in mysql_result0.get_data_on_results()]
        if 'ongoing' in task_status_list:
            return Resp.build_error(
                code=50001,
                message="当前由于计算资源，每个用户单次只能执行一个任务"
            )

        # 删除图片文件
        path_to_delete = cls.TIME_EVOLVE_IMAGE_PATH.format(user_id, task_id).rstrip("/image")
        if os.path.exists(path_to_delete):
            try:
                shutil.rmtree(path_to_delete)
            except Exception as e:
                print(e)

        # 删除 neo4j 数据库中的数据
        neo4j_result = ScienceResearchMapper.delete_science_paper_and_science_theme_node({
            "task_id": task_id
        })

        # 删除 sr_graph 表中的数据
        mysql_result = ScienceResearchMapper.delete_graph({
            "task_id": task_id
        })

        # 删除 sr_theme_explaining 表中的数据
        mysql_result1 = ScienceResearchMapper.delete_theme_explaining({
            "task_id": task_id
        })

        # 删除 sr_theme_higher_explaining 表中的数据
        mysql_result2 = ScienceResearchMapper.delete_theme_higher_explaining({
            "task_id": task_id
        })

        # 删除 sr_theme_translate 表中的数据
        mysql_result3 = ScienceResearchMapper.delete_theme_translate({
            "task_id": task_id
        })

        # 删除 sr_theme_characterize 表中的数据
        mysql_result5 = ScienceResearchMapper.delete_theme_characterize({
            "task_id": task_id
        })

        mysql_result6 = ScienceResearchMapper.select_paper_search_where_user_id_and_task_id({
            "user_id": user_id,
            "task_id": task_id
        })

        paper_search = mysql_result6.get_data_on_results()[0]

        search_words = paper_search['search_words']
        paper_num = paper_search['paper_num']
        task_name = paper_search['task_name']
        task_type = paper_search['task_type']
        pdf_path = paper_search['pdf_path']

        search_id_list = [record['search_id'] for record in mysql_result6.get_data_on_results()]

        for search_id in search_id_list:
            # 修改任务状态
            mysql_result4 = ScienceResearchMapper.update_paper_search_set_task_status({
                "task_status": "ongoing",
                "search_id": search_id
            })

        time_evolve_image_path = cls.TIME_EVOLVE_IMAGE_PATH.format(user_id, task_id)
        mermaid_image_path = cls.MERMAID_IMAGE_PATH.format(user_id, task_id)

        # 初始化图像文件保存根路径
        if not os.path.exists(time_evolve_image_path):
            os.makedirs(time_evolve_image_path)

        knowledge_finding_config = {
            "search_words": search_words,
            "paper_num": paper_num,
            "paper_pdf_path": pdf_path,
            "user_id": user_id,
            "task_id": task_id,
            "task_name": task_name,
            "task_type": task_type,
            "search_id_list": search_id_list,
            "time_evolve_image_path": time_evolve_image_path,
            "mermaid_image_path": mermaid_image_path
        }

        # 创建进度
        ScienceResearchProgress.create(user_id=user_id, task_id=task_id, task_name=task_name)

        knowledge_finding = KnowledgeFindingTask(knowledge_finding_config)
        knowledge_finding_buffer_result = KnowledgeGraphTaskBuffer.start_running(
            method=lambda: knowledge_finding.start_create_knowledge_graph(),
            info=knowledge_finding_config
        )

        return Resp.build_success(
            code=knowledge_finding_buffer_result.code,
            message=knowledge_finding_buffer_result.message
        )

    @classmethod
    def get_task_progress(cls, token):

        # 解析token
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        task_progress = ScienceResearchProgress.get(user_id=user_id)

        return Resp.build_success(data={
            "taskProgress": task_progress
        })

    @classmethod
    def task_delete_task(cls, task_id, token):

        # 解析token
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        # 如果任务正在进行中，无法删除
        mysql_result0 = ScienceResearchMapper.select_paper_search_where_user_id_and_task_id({
            "user_id": user_id,
            "task_id": task_id
        })

        task_status_list = [record['task_status'] for record in mysql_result0.get_data_on_results()]
        if 'ongoing' in task_status_list:
            return Resp.build_error(
                code=50001,
                message="任务正在进行中，无法删除"
            )

        # 删除图片文件
        path_to_delete = cls.TIME_EVOLVE_IMAGE_PATH.format(user_id, task_id).rstrip("/image")
        if os.path.exists(path_to_delete):
            try:
                shutil.rmtree(path_to_delete)
            except Exception as e:
                print(e)

        # 删除 neo4j 数据库中的数据
        neo4j_result = ScienceResearchMapper.delete_science_paper_and_science_theme_node({
            "task_id": task_id
        })

        # 删除 sr_graph 表中的数据
        mysql_result = ScienceResearchMapper.delete_graph({
            "task_id": task_id
        })

        # 删除 sr_theme_explaining 表中的数据
        mysql_result1 = ScienceResearchMapper.delete_theme_explaining({
            "task_id": task_id
        })

        # 删除 sr_theme_higher_explaining 表中的数据
        mysql_result2 = ScienceResearchMapper.delete_theme_higher_explaining({
            "task_id": task_id
        })

        # 删除 sr_theme_translate 表中的数据
        mysql_result3 = ScienceResearchMapper.delete_theme_translate({
            "task_id": task_id
        })

        # 删除 sr_theme_filtered 表中的数据
        mysql_result4 = ScienceResearchMapper.delete_theme_filtered({
            "task_id": task_id
        })

        # 删除 sr_theme_characterize 表中的数据
        mysql_result5 = ScienceResearchMapper.delete_theme_characterize({
            "task_id": task_id
        })

        # 删除 sr_paper_search 表中的数据
        mysql_result6 = ScienceResearchMapper.delete_paper_search({
            "task_id": task_id
        })

        return Resp.build_success()

    @classmethod
    def start_create_knowledge_graph(cls, task_params, task_type, token):

        # 解析token
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        # 每个用户单次只能执行一个任务
        mysql_result0 = ScienceResearchMapper.select_paper_search_where_user_id({
            "user_id": user_id
        })

        task_status_list = [record['task_status'] for record in mysql_result0.get_data_on_results()]
        if 'ongoing' in task_status_list:
            return Resp.build_error(
                code=50001,
                message="当前版本由于计算资源，每个用户单次只能执行一个任务"
            )

        # 初始化论文pdf文件保存根路径
        if not os.path.exists(cls.PAPER_PDF_PATH):
            os.makedirs(cls.PAPER_PDF_PATH)

        if task_type == 'retrieval':
            search_words = ""
            # 默认值
            paper_num = 20
            task_name = ""
            for param in task_params:
                if param['name'] == 'fieldKeywords':
                    search_words = param['value']
                elif param['name'] == 'paperNum':
                    paper_num = int(param['value'])
                elif param['name'] == 'taskName':
                    task_name = param['value']

            paper_pdf_path = cls.PAPER_PDF_PATH

            # 检索式任务论文最大数量限制
            max_paper_num = 130
            if paper_num > max_paper_num:
                return Resp.build_error(
                    code=50001,
                    message=f"检索式任务设定论文数量过大，请重新设定 (不能超过{max_paper_num}篇)"
                )

        else:
            search_words = ""
            folder_file_name = ""
            task_name = ""
            for param in task_params:
                if param['name'] == 'fieldKeywords':
                    search_words = param['value']
                elif param['name'] == 'folderFileId':
                    folder_file_name = param['value']
                elif param['name'] == 'taskName':
                    task_name = param['value']

            # 获取文件夹文件id
            mysql_result1 = ScienceResearchMapper.select_file_where_file_name({
                "file_name": folder_file_name
            })
            file_result_list = mysql_result1.get_data_on_results()

            if len(file_result_list) == 0:
                return Resp.build_error(
                    code=50002,
                    message="解析式任务中参数[检索论文pdf文件夹名称]不存在，请检查并重试"
                )

            folder_file_id = file_result_list[0]['file_id']

            paper_pdf_path = cls.FILE_PATH.format(user_id, folder_file_id)

            paper_num = len([x for x in os.listdir(paper_pdf_path)])

        # 生成任务id
        task_id = RandomStrGenerator.generate_uuid()

        time_evolve_image_path = cls.TIME_EVOLVE_IMAGE_PATH.format(user_id, task_id)
        mermaid_image_path = cls.MERMAID_IMAGE_PATH.format(user_id, task_id)

        # 初始化时间演化图像文件保存根路径
        if not os.path.exists(time_evolve_image_path):
            os.makedirs(time_evolve_image_path)

        # 初始化流程图图像文件保存根路径
        if not os.path.exists(mermaid_image_path):
            os.makedirs(mermaid_image_path)

        search_id_list = []
        for _ in range(paper_num):
            # 生成查询id
            search_id = RandomStrGenerator.generate_uuid()
            search_id_list.append(search_id)

            mysql_result = ScienceResearchMapper.insert_paper_search({
                "search_id": search_id,
                "user_id": user_id,
                "task_id": task_id,
                "task_name": task_name,
                "search_words": search_words,
                "paper_num": paper_num,
                "task_type": task_type,
                "pdf_path": paper_pdf_path
            })

        knowledge_finding_config = {
            "search_words": search_words,
            "paper_num": paper_num,
            "paper_pdf_path": paper_pdf_path,
            "user_id": user_id,
            "task_id": task_id,
            "task_name": task_name,
            "task_type": task_type,
            "search_id_list": search_id_list,
            "time_evolve_image_path": time_evolve_image_path,
            "mermaid_image_path": mermaid_image_path
        }

        # 创建进度
        ScienceResearchProgress.create(user_id=user_id, task_id=task_id, task_name=task_name)

        knowledge_finding = KnowledgeFindingTask(knowledge_finding_config)
        knowledge_finding_buffer_result = KnowledgeGraphTaskBuffer.start_running(
            method=lambda: knowledge_finding.start_create_knowledge_graph(),
            info=knowledge_finding_config
        )

        return Resp.build_success(
            code=knowledge_finding_buffer_result.code,
            message=knowledge_finding_buffer_result.message
        )

    @classmethod
    def get_knowledge_graph_node_detail_info(cls, node_id, token):

        mysql_result = ScienceResearchMapper.select_paper_where_paper_id({
            "paper_id": node_id
        })

        paper_obj = mysql_result.get_data_on_results()[0]

        research_questions = {}
        research_methods = {}
        research_contributions = {}
        research_limitations = {}
        try:
            research_questions = eval(paper_obj['research_questions'])
            research_methods = eval(paper_obj['research_methods'])
            research_contributions = eval(paper_obj['research_contributions'])
            research_limitations = eval(paper_obj['research_limitations'])

        except Exception as e:
            print(e)

        mysql_resul1 = ScienceResearchMapper.select_theme_translate({})

        zh_theme_name_mapping = {record['theme_name']: record['theme_zh_name'] for record in mysql_resul1.get_data_on_results()}

        research_questions = [{
            "name": name,
            "value": value,
            "zh_name": zh_theme_name_mapping.get(name, "")
        } for name, value in research_questions.items()]

        research_methods = [{
            "name": name,
            "value": value,
            "zh_name": zh_theme_name_mapping.get(name, "")
        } for name, value in research_methods.items()]

        research_contributions = [{
            "name": name,
            "value": value,
            "zh_name": zh_theme_name_mapping.get(name, "")
        } for name, value in research_contributions.items()]

        research_limitations = [{
            "name": name,
            "value": value,
            "zh_name": zh_theme_name_mapping.get(name, "")
        } for name, value in research_limitations.items()]

        node_detail_info = {
            "paper_id": paper_obj['paper_id'],
            "paper_title": paper_obj['paper_title'],
            "paper_zh_title": paper_obj['paper_zh_title'],
            "paper_doi": paper_obj['paper_doi'],
            "paper_authors": paper_obj['paper_authors'],
            "paper_publish_time": paper_obj['paper_publish_time'],
            "paper_journal": paper_obj['paper_journal'],
            "paper_abstract": paper_obj['paper_abstract'],
            "paper_zh_abstract": paper_obj['paper_zh_abstract'],
            "paper_url": paper_obj['paper_url'],
            "research_questions": research_questions,
            "research_methods": research_methods,
            "research_contributions": research_contributions,
            "research_limitations": research_limitations,
            "paper_summary": paper_obj['paper_summary'],
            "paper_zh_summary": paper_obj['paper_zh_summary']
        }

        return Resp.build_success(data={
            "nodeDetailInfo": node_detail_info
        })

    @classmethod
    def get_content_perspective_dropdown(cls):

        content_perspective_dropdown = [
            {
                "id": 0,
                "label": "研究问题",
                "name": "research_questions",
            },
            {
                "id": 1,
                "label": "研究方法",
                "name": "research_methods",
            },
            {
                "id": 2,
                "label": "研究贡献",
                "name": "research_contributions",
            },
            {
                "id": 3,
                "label": "研究局限",
                "name": "research_limitations",
            },
        ]

        return Resp.build_success(data={
            "contentPerspectiveOptions": content_perspective_dropdown
        })

    @classmethod
    def get_knowledge_network(cls, task_id, content_perspective):

        # 获取所有主题词节点
        neo4j_result = ScienceResearchMapper.match_all_theme_node_on_root_id({
            "root_id": task_id
        }, content_perspective)
        if not neo4j_result.check:
            return Resp.build_db_error()

        theme_list = neo4j_result.get_data_on_results()

        # 获取所有论文节点
        neo4j_result1 = ScienceResearchMapper.match_all_paper_node_on_root_id({
            "root_id": task_id
        }, content_perspective)
        if not neo4j_result1.check:
            return Resp.build_db_error()

        paper_list = neo4j_result1.get_data_on_results()

        for i in range(len(theme_list)):
            if theme_list[i]['theme_explaining'] is None:
                theme_list[i]['theme_explaining'] = ""

        for i in range(len(paper_list)):
            if not hasattr(paper_list[i], 'theme_explaining'):
                paper_list[i]['theme_explaining'] = ""

        # 获取所有主题词的翻译
        mysql_resul1 = ScienceResearchMapper.select_theme_translate({})

        zh_theme_name_mapping = {record['theme_name'].lower(): record['theme_zh_name'] for record in mysql_resul1.get_data_on_results()}

        for i in range(len(theme_list)):
            theme_list[i]['zh_label'] = zh_theme_name_mapping.get(theme_list[i]['label'].lower(), "")

        # 获取所有论文的翻译
        mysql_resul2 = ScienceResearchMapper.select_paper({})

        zh_paper_name_mapping = {record['paper_title'].lower(): record['paper_zh_title'] for record in mysql_resul2.get_data_on_results()}

        for i in range(len(paper_list)):
            paper_list[i]['zh_label'] = zh_paper_name_mapping.get(paper_list[i]['label'].lower(), "")

        # 隐藏没有论文节点关联的主题词节点
        filtered_root_list = []
        for theme in theme_list:
            if len(theme['children_ids']) > 1:
                filtered_root_list.append(theme)
            else:
                print(f"丢失节点: {theme['label']}")
        theme_list = filtered_root_list

        # 获取所有边
        neo4j_result2 = ScienceResearchMapper.match_all_edge_on_root_id({
            "root_id": task_id
        }, content_perspective)
        if not neo4j_result2.check:
            return Resp.build_db_error()

        edge_list = neo4j_result2.get_data_on_results()
        edge_dict = {}
        for edge in edge_list:
            if edge['target'] not in edge_dict.keys():
                edge_dict[edge['target']] = edge['weight']

        for i in range(len(theme_list)):
            if theme_list[i]['id'] in edge_dict.keys():
                theme_list[i]['weight'] = edge_dict[theme_list[i]['id']]
            else:
                theme_list[i]['weight'] = 1.0
        for i in range(len(paper_list)):
            if paper_list[i]['id'] in edge_dict.keys():
                paper_list[i]['weight'] = edge_dict[paper_list[i]['id']]
            else:
                paper_list[i]['weight'] = 1.0

        # 递归获取树结构
        graph_data = GraphParser.science_init_recursive_get_tree(
            root_list=theme_list,
            paper_list=paper_list,
        )

        theme_explaining_zh_name_mapping = {
            "research_questions": "research_zh_questions",
            "research_methods": "research_zh_methods",
            "research_contributions": "research_zh_contributions",
            "research_limitations": "research_zh_limitations"
        }

        # 获取高表征主题词解释的翻译
        mysql_resul3 = ScienceResearchMapper.select_theme_higher_explaining({
            "task_id": task_id
        })

        higher_theme_explaining_mapping = {}
        try:
            higher_theme_explaining_mapping = eval([record for record in mysql_resul3.get_data_on_results()][0][theme_explaining_zh_name_mapping[content_perspective]])

            higher_theme_explaining_mapping = {k.lower(): v for k, v in higher_theme_explaining_mapping.items()}
        except Exception as e:
            print(e)

        # 获取主题词解释的翻译
        mysql_resul4 = ScienceResearchMapper.select_theme_explaining({
            "task_id": task_id
        })

        theme_explaining_mapping = {}
        try:
            theme_explaining_mapping = eval([record for record in mysql_resul4.get_data_on_results()][0][theme_explaining_zh_name_mapping[content_perspective]])

            theme_explaining_mapping = {k.lower(): v for k, v in theme_explaining_mapping.items()}
        except Exception as e:
            print(e)

        for i in range(len(graph_data['children'])):
            higher_keyword = graph_data['children'][i]['label'].split('*')[0].strip(' ').lower()
            graph_data['children'][i]['theme_explaining'] = f"{graph_data['children'][i]['theme_explaining']}    *{higher_theme_explaining_mapping.pop(higher_keyword, '')}"

            current_theme_explaining_mapping = theme_explaining_mapping[higher_keyword]
            current_theme_explaining_mapping = {k.lower(): v for k, v in current_theme_explaining_mapping.items()}
            for j in range(len(graph_data['children'][i]['children'])):
                keyword = graph_data['children'][i]['children'][j]['label'].split('*')[0].strip(' ').lower()
                graph_data['children'][i]['children'][j]['theme_explaining'] = f"{graph_data['children'][i]['children'][j]['theme_explaining']}    *{current_theme_explaining_mapping.pop(keyword, '')}"

        # 每个层级按照权重排序
        new_children_list = []
        for higher_keyword_node in graph_data['children']:
            higher_keyword_node['children'] = sorted(higher_keyword_node['children'], key=lambda v: v['weight'], reverse=True)
            new_children_list.append(higher_keyword_node)

        new_children_list = sorted(new_children_list, key=lambda v: v['weight'], reverse=True)

        graph_data['children'] = new_children_list

        return Resp.build_success(data={
            "graphData": graph_data
        })

    @classmethod
    def get_paper_communities(cls, token):

        # 解析token
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        mysql_result = ScienceResearchMapper.select_community({
            "node_owner": user_id
        })

        node_community_list = [x.to_dict() for x in mysql_result.get_data_on_results()]

        community_index_list = list(set([x['node_belong_community'] for x in node_community_list]))

        node_community_dict = dict(zip(community_index_list, [[] for _ in community_index_list]))
        for node in node_community_list:
            node_community_dict[node['node_belong_community']].append(node)

        paper_community_list = list(node_community_dict.values())
        separated_paper_community_list = []
        for x in paper_community_list:
            paper_list = []
            theme_list = []
            for y in x:
                if y['node_label'] == 'science_paper':
                    paper_list.append(y)
                elif y['node_label'] == 'science_theme':
                    theme_list.append(y)
            separated_paper_community_list.append({
                "paper_list": paper_list,
                "theme_list": theme_list
            })

        return Resp.build_success(data={
            "paperCommunityList": separated_paper_community_list
        })

    @classmethod
    def cluster_papers(cls, task_id, token):

        # 解析token
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        # 查询所有知识网络中的节点 (目前去掉检索主题词节点)
        neo4j_result = ScienceResearchMapper.match_all_science_nodes({})
        nodes_list = [x for x in neo4j_result.get_data_on_results()]

        detected_nodes = []
        detected_edges = []
        detected_node_names = {}
        detected_node_labels = {}
        for record in nodes_list:
            source = record["source"]
            target = record["target"]
            weight = float(record["weight"])

            detected_nodes.append(source)
            detected_nodes.append(target)

            detected_edges.append((source, target, weight))

            detected_node_names[source] = record['source_name']
            detected_node_names[target] = record['target_name']

            detected_node_labels[source] = record['source_labels'][0] if isinstance(record['source_labels'], list) and len(record['source_labels']) > 0 else record['source_labels']
            detected_node_labels[target] = record['target_labels'][0] if isinstance(record['target_labels'], list) and len(record['target_labels']) > 0 else record['target_labels']

        detected_nodes = list(set(detected_nodes))

        # # 转换图结构为链接结构
        # out_edges = {}
        # for record in nodes_list:
        #     source = record["source"]
        #     target = record["target"]
        #     weight = float(record["weight"])  # 确保转换为浮点数
        # 
        #     if target not in out_edges:
        #         out_edges[target] = {}
        #     out_edges[target][source] = weight
        # 
        # # 计算知识网络节点之间的相似度
        # similarity = SimilarityCalculation.weighted_reverse_simrank(
        #     out_edges=out_edges
        # )
        # 
        # # 将相似度转换为矩阵格式
        # all_nodes = [x for x in similarity.keys()]
        # zero_df = np.zeros((len(all_nodes), len(all_nodes)))
        # similarity_df = pd.DataFrame(
        #     data=zero_df,
        #     columns=all_nodes,
        #     index=all_nodes
        # )
        # 
        # for source_id, target_node in similarity.items():
        #     similarity_df.loc[source_id] = target_node
        # 
        # # 获取所有的论文节点id
        # cypher_result1 = ScienceResearchMapper.match_science_paper_node({})
        # detected_nodes = [record['n']['id'] for record in cypher_result1.get_data_on_results()]
        # 
        # paper_similarity_df = similarity_df.loc[detected_nodes, detected_nodes]
        # 
        # # 转换图结构为集合结构
        # detected_edges = []
        # for source_paper_id in detected_nodes:
        #     for target_paper_id in paper_similarity_df.columns:
        #         detected_edges.append((source_paper_id, target_paper_id, paper_similarity_df.loc[source_paper_id, target_paper_id]))

        community_result = CommunityDetection.infomap(
            nodes=detected_nodes,
            edges=detected_edges
        )

        # 打印结果示例
        print(f"检测到 {len(community_result)} 个社区")
        for i, comm in enumerate(community_result[:3], 1):
            print(f"社区{i} (共{len(comm)}个节点): {comm[:3]}...")

        community_dict = {}
        for i, community in enumerate(community_result):
            for node in community:
                community_dict[node] = i

        no_weight_edges = [(x[0], x[1]) for x in detected_edges]

        # 图聚类可视化
        DataVisualize.draw_community_graph(
            nodes=detected_nodes,
            edges=no_weight_edges,
            communities=community_dict,
            name_mapping=detected_node_names
        )

        for i, community in enumerate(community_result):
            for node_id in community:
                sql_result = ScienceResearchMapper.insert_community({
                    "node_id": node_id,
                    "node_label": detected_node_labels[node_id],
                    "node_name": detected_node_names[node_id],
                    "node_owner": user_id,
                    "node_belong_community": i
                })

        return Resp.build_success()

    @classmethod
    def reconstruct_graph(cls, task_id, token):

        # 解析token
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        # 查找高表征主题词的数据
        mysql_result = ScienceResearchMapper.select_paper_higher_summary_where_user_id({
            "user_id": user_id
        })

        paper_higher_summary_list = mysql_result.get_data_on_results()
        if len(paper_higher_summary_list) == 0:
            return Resp.build_error()

        paper_higher_summary = mysql_result.get_data_on_results()[0].to_dict()

        knowledge_finding_config = {
            "paper_higher_summary": paper_higher_summary,
            "user_id": user_id
        }

        knowledge_finding = KnowledgeFindingTask(knowledge_finding_config)
        knowledge_finding_buffer_result = KnowledgeGraphTaskBuffer.start_running(method=lambda: knowledge_finding.create_graph())

        return Resp.build_success(
            code=knowledge_finding_buffer_result.code,
            message=knowledge_finding_buffer_result.message
        )

    @classmethod
    def characterize_themes(cls, task_id, token):

        # 解析token
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        mysql_result = ScienceResearchMapper.select_paper_summaries_where_user_id({
            "paper_owner": user_id
        })

        paper_summary_list = mysql_result.get_data_on_results()

        knowledge_finding_config = {
            "paper_summary_list": paper_summary_list,
            "user_id": user_id
        }

        knowledge_finding = KnowledgeFindingTask(knowledge_finding_config)
        knowledge_finding_buffer_result = KnowledgeGraphTaskBuffer.start_running(method=lambda: knowledge_finding.characterize_themes())

        return Resp.build_success(
            code=knowledge_finding_buffer_result.code,
            message=knowledge_finding_buffer_result.message
        )

    @classmethod
    def get_paper_summaries(cls, token):

        # 解析token
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        mysql_result = ScienceResearchMapper.select_paper_where_user_id({
            "user_id": user_id
        })

        paper_list = mysql_result.get_data_on_results()

        mysql_result1 = ScienceResearchMapper.select_paper_summaries_where_user_id({
            "user_id": user_id
        })

        paper_summary_list = mysql_result1.get_data_on_results()

        merged_paper_summary_dict = {}
        for paper_summary in paper_summary_list:
            merged_paper_summary_dict[paper_summary.paper_id] = paper_summary.to_dict()

        for paper in paper_list:
            if paper.paper_id in merged_paper_summary_dict.keys():
                merged_paper_summary_dict[paper.paper_id].update(paper.to_dict())

        merged_paper_summary_list = list(merged_paper_summary_dict.values())

        return Resp.build_success(data={
            "paperSummaryList": merged_paper_summary_list
        })

    @classmethod
    def generate_report(cls, task_id, token):

        # 解析token
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        mysql_result = ScienceResearchMapper.select_paper_summaries_where_user_id({
            "user_id": user_id
        })

        paper_summary_list = mysql_result.get_data_on_results()

        paper_summary_str_list = [f"""
            research_questions: {paper_summary.research_questions}, 
            research_methods: {paper_summary.research_methods}
        """ for paper_summary in paper_summary_list]

        knowledge_finding_config = {
            "paper_summary_str_list": paper_summary_str_list
        }

        knowledge_finding = KnowledgeFindingTask(knowledge_finding_config)
        knowledge_finding_buffer_result = KnowledgeGraphTaskBuffer.start_running(method=lambda: knowledge_finding.generate_report())

        return Resp.build_success(
            code=knowledge_finding_buffer_result.code,
            message=knowledge_finding_buffer_result.message
        )

    @classmethod
    def get_my_space_tasks(cls, token):

        # 解析token
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        mysql_result = ScienceResearchMapper.select_paper_search_where_user_id({
            "user_id": user_id
        })

        paper_search_list = mysql_result.get_data_on_results()
        task_list = list(set([record['task_id'] for record in paper_search_list]))
        task_dict = dict(zip(task_list, [{} for _ in range(len(task_list))]))

        for paper_search in paper_search_list:
            if 'user_id' not in task_dict[paper_search['task_id']].keys():
                task_dict[paper_search['task_id']]['user_id'] = paper_search['user_id']
            if 'task_id' not in task_dict[paper_search['task_id']].keys():
                task_dict[paper_search['task_id']]['task_id'] = paper_search['task_id']
            if 'task_name' not in task_dict[paper_search['task_id']].keys():
                task_dict[paper_search['task_id']]['task_name'] = paper_search['task_name']
            if 'search_words' not in task_dict[paper_search['task_id']].keys():
                task_dict[paper_search['task_id']]['search_words'] = paper_search['search_words']
            if 'create_timestamp' not in task_dict[paper_search['task_id']].keys():
                task_dict[paper_search['task_id']]['create_timestamp'] = str(TimeParser.convert_time_format(paper_search['create_timestamp']))
            if 'task_status' not in task_dict[paper_search['task_id']].keys():
                task_dict[paper_search['task_id']]['task_status'] = paper_search['task_status']

        my_space_tasks = sorted([x for x in task_dict.values()], key=lambda v: v['create_timestamp'], reverse=True)

        my_space_tasks_headers = [
            {
                "id": 0,
                "label": '任务名',
                "name": 'task_name'
            },
            {
                "id": 1,
                "label": '检索关键词',
                "name": 'search_words'
            },
            {
                "id": 2,
                "label": '创建时间',
                "name": 'create_timestamp'
            },
            {
                "id": 3,
                "label": '任务状态',
                "name": 'task_status'
            },
        ]

        return Resp.build_success(data={
            "mySpaceTasks": my_space_tasks,
            "mySpaceTasksHeaders": my_space_tasks_headers
        })

    @classmethod
    def summarize_paper_pdf(cls, task_id, token):

        # 解析token
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        base_path = cls.PAPER_PDF_PATH.format(user_id, task_id)

        pdf_paper_path = f"{base_path}/pdf_paper"

        knowledge_finding_config = {
            "base_path": base_path,
            "pdf_paper_path": pdf_paper_path,
            "user_id": user_id,
            "task_id": task_id
        }

        knowledge_finding = KnowledgeFindingTask(knowledge_finding_config)
        knowledge_finding_buffer_result = KnowledgeGraphTaskBuffer.start_running(method=lambda: knowledge_finding.summarize_paper_pdf())

        return Resp.build_success(
            code=knowledge_finding_buffer_result.code,
            message=knowledge_finding_buffer_result.message
        )

    @classmethod
    def download_paper_pdf(cls, task_id, user_id):

        # 初始化论文pdf文件保存根路径
        if not os.path.exists(cls.PAPER_PDF_PATH):
            os.makedirs(cls.PAPER_PDF_PATH)

        knowledge_finding_config = {
            "pdf_paper_path": cls.PAPER_PDF_PATH,
            "user_id": user_id,
            "task_id": task_id
        }

        knowledge_finding = KnowledgeFindingTask(knowledge_finding_config)
        knowledge_finding_buffer_result = KnowledgeGraphTaskBuffer.start_running(method=lambda: knowledge_finding.download_paper_pdf())

        return Resp.build_success(
            code=knowledge_finding_buffer_result.code,
            message=knowledge_finding_buffer_result.message
        )

    @classmethod
    def search_papers_on_themes_zh(cls, search_words, token):

        # 解析token
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        task_id = RandomStrGenerator.generate_uuid()

        base_path = cls.PAPER_PDF_PATH.format(user_id, task_id)
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        knowledge_finding_config = {
            "search_words": search_words,
            "base_path": base_path,
            "user_id": user_id,
            "task_id": task_id
        }

        knowledge_finding = KnowledgeFindingTask(knowledge_finding_config)
        knowledge_finding_buffer_result = KnowledgeGraphTaskBuffer.start_running(
            method=lambda: knowledge_finding.search_papers_on_themes_zh())

        return Resp.build_success(
            code=knowledge_finding_buffer_result.code,
            message=knowledge_finding_buffer_result.message
        )

    @classmethod
    def search_papers_on_themes(cls, search_words, paper_num, user_id):

        # 创建任务id
        task_id = RandomStrGenerator.generate_uuid()

        knowledge_finding_config = {
            "search_words": search_words,
            "paper_num": paper_num,
            "user_id": user_id,
            "task_id": task_id
        }

        knowledge_finding = KnowledgeFindingTask(knowledge_finding_config)
        knowledge_finding_buffer_result = KnowledgeGraphTaskBuffer.start_running(method=lambda: knowledge_finding.search_papers_on_themes())

        return Resp.build_success(
            code=knowledge_finding_buffer_result.code,
            message=knowledge_finding_buffer_result.message,
            data={
                "task_id": task_id
            }
        )





