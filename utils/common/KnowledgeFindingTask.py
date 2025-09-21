import os
import time
import re
import requests

from algorithm.ClusterClassification import ClusterClassification
from algorithm.DistanceCalculation import DistanceCalculation
from mapper.KnowledgeBaseMapper import KnowledgeBaseMapper
from mapper.ScienceResearchMapper import ScienceResearchMapper
from buffer.ScienceResearchProgress import ScienceResearchProgress
from utils.common.EmailSender import EmailSender
from utils.common.GraphParser import GraphParser
from utils.common.HashParser import HashParser
from utils.common.PDFParser import PDFParser
from utils.common.RandomStrGenerator import RandomStrGenerator
from utils.common.RequestSender import RequestSender
from utils.common.SemanticVectorCalculation import SemanticVectorCalculation
from utils.common.TextTranslator import TextTranslator
from utils.llm.APILLMParser import APILLMParser
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class KnowledgeFindingTask:
    def __init__(self, knowledge_finding_config):
        self.knowledge_finding_config = knowledge_finding_config

    def start_create_knowledge_graph(self):

        user_id = self.knowledge_finding_config['user_id']

        # 更新进度
        ScienceResearchProgress.update_current_stage(user_id=user_id, new_stage="根据关键词获取论文中...")

        # 根据关键词获取论文
        self.search_papers_on_themes()

        # 更新进度
        ScienceResearchProgress.update_total_progress(user_id=user_id, adding=10)
        ScienceResearchProgress.update_current_stage(user_id=user_id, new_stage="下载论文pdf中...")

        # 下载论文pdf
        self.download_paper_pdf()

        # 更新进度
        ScienceResearchProgress.update_total_progress(user_id=user_id, adding=10)
        ScienceResearchProgress.update_current_stage(user_id=user_id, new_stage="根据论文总结生成主题词中...")

        # 对任务所有的论文总结生成关键词
        self.summarize_paper_themes()

        # 更新进度
        ScienceResearchProgress.update_total_progress(user_id=user_id, adding=10)
        ScienceResearchProgress.update_current_stage(user_id=user_id, new_stage="根据主题词生成高表征主题词中...")

        # 根据所有的主题词生成高表征主题词
        self.characterize_themes()

        # 更新进度
        ScienceResearchProgress.update_total_progress(user_id=user_id, adding=20)
        ScienceResearchProgress.update_current_stage(user_id=user_id, new_stage="生成知识网络图结构中...")

        # 生成知识网络图结构
        self.create_graph()

        # 更新进度
        ScienceResearchProgress.update_total_progress(user_id=user_id, adding=10)
        ScienceResearchProgress.update_current_stage(user_id=user_id, new_stage="重构知识网络图结构中...")

        # 更新进度
        ScienceResearchProgress.update_total_progress(user_id=user_id, adding=10)
        ScienceResearchProgress.update_current_stage(user_id=user_id, new_stage="翻译主题词中...")

        # 翻译主题词
        self.translate_theme_keywords()

        # 更新进度
        ScienceResearchProgress.update_total_progress(user_id=user_id, adding=20)
        ScienceResearchProgress.update_current_stage(user_id=user_id, new_stage="生成文献综述报告中...")

        # 生成文献综述报告
        self.summary_knowledge_graph()

        # 更新进度
        ScienceResearchProgress.update_total_progress(user_id=user_id, adding=10)
        ScienceResearchProgress.update_current_stage(user_id=user_id, new_stage="生成时间演化趋势图中...")

        # # 生成时间演化趋势图
        # self.draw_time_evolve_wordcloud()

        # 更新进度
        ScienceResearchProgress.update_total_progress(user_id=user_id, adding=10)
        ScienceResearchProgress.update_current_stage(user_id=user_id, new_stage="任务已完成")

        # 获取用户邮箱
        mysql_result = KnowledgeBaseMapper.select_user({
            "user_id": user_id
        })

        user_email = mysql_result.get_data_on_results()[0]['user_email']

        # 发送任务已完成信息到用户邮箱
        try:
            EmailSender.send_knowledge_graph_task_success(
                email=user_email,
                task_id=self.knowledge_finding_config['task_id'],
                task_name=self.knowledge_finding_config['task_name']
            )
        except Exception as e:
            print(e)

    def translate_theme_keywords(self):

        ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'], new_message="开始翻译")

        # 获取所有主题词节点
        to_translate_name_list = []
        for content_perspective in ['research_questions', 'research_methods', 'research_contributions', 'research_limitations']:
            neo4j_result = ScienceResearchMapper.match_all_theme_node_on_root_id({
                "root_id": self.knowledge_finding_config['task_id']
            }, content_perspective)

            theme_list = neo4j_result.get_data_on_results()
            to_translate_name_list += [x['label'] for x in theme_list]

        to_translate_name_list = [x for x in list(set(to_translate_name_list)) if x and x not in ["", " "]]

        # 分割列表为段落

        paragraph_list = []
        paragraph_len_list = []
        max_str = 600
        counter = 0
        current_paragraph = ""
        for name in to_translate_name_list:
            name = name.replace(",", " ")
            if len(current_paragraph) + len(name) > max_str:
                paragraph_list.append(current_paragraph.rstrip(", "))
                paragraph_len_list.append(counter)
                current_paragraph = ""
                counter = 0

            current_paragraph += name + ", "
            counter += 1
        else:
            paragraph_list.append(current_paragraph.rstrip(", "))
            paragraph_len_list.append(counter)

        example_input = """
                        Data treatment inconsistency, Hybrid modeling approaches, Grease degradation modeling, Adaptive learning in wind turbine fault diagnosis, Mooring system failure effects
                    """

        example_output = """
                        数据处理不一致性，混合建模方法，润滑脂降解建模，风电机组故障诊断中的自适应学习，泊系统故障影响
                    """

        # 翻译检索词
        translated_search_words = TextTranslator.translate(self.knowledge_finding_config['search_words'], translate_source="deepseek").get_data_on_results()

        translated_name_dict = {}
        total_translated_list = []
        for m, paragraph in enumerate(paragraph_list):
            translated_list = [" " for _ in range(len(paragraph.split(",")))]
            max_retires = 10
            for _ in range(max_retires):
                try:
                    translated_paragraph = APILLMParser.call_llm_api(
                        system_prompt=f"【{translated_search_words}领域专业】将用户给出的英文翻译为中文，输出保持与输入相同的格式，输出仅含译文，不要含任何注释说明内容。 {{示例输入: {example_input}}}， {{示例输出: {example_output}}}",
                        user_prompt=paragraph
                    )

                    # 使用正则表达式去除括号及括号内的内容
                    translated_paragraph = re.sub(r'（注：[^）]*）', '', translated_paragraph)

                    translated_list = [x for x in re.split(r"[，,]+", translated_paragraph) if x and x not in ["", " "]]

                    if len(translated_list) == paragraph_len_list[m]:
                        break

                except Exception as e:
                    print(e)

            else:
                pass

            total_translated_list = total_translated_list + translated_list

        translated_name_dict = dict(zip(to_translate_name_list, total_translated_list))

        # 保存主题词翻译到数据库
        for theme_name, theme_zh_name in translated_name_dict.items():
            if theme_name in ["", " "] or theme_zh_name in ["", " "]:
                continue

            mysql_result2 = ScienceResearchMapper.select_theme_translate_where_theme_name({
                "theme_name": theme_name
            })

            # 如果不存在该关键词的翻译
            if not mysql_result2.verify_data_on_results():
                theme_id = RandomStrGenerator.generate_uuid()

                mysql_result3 = ScienceResearchMapper.insert_theme_translate({
                    "theme_id": theme_id,
                    "theme_name": theme_name,
                    "theme_zh_name": theme_zh_name,
                    "user_id": self.knowledge_finding_config['user_id'],
                    "task_id": self.knowledge_finding_config['task_id']
                })

        ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'], new_message="所有主题词已翻译完成")

        # 翻译主题词解释句子
        mysql_result4 = ScienceResearchMapper.select_theme_explaining({
            "task_id": self.knowledge_finding_config['task_id']
        })

        higher_theme_explaining_obj = mysql_result4.get_data_on_results()[0]

        translated_higher_theme_explaining_obj = {}
        for content_perspective in ['research_questions', 'research_methods', 'research_contributions', 'research_limitations']:
            higher_theme_dict = eval(higher_theme_explaining_obj[content_perspective])
            translated_higher_theme_dict = {}
            for higher_keyword, group_theme_dict in higher_theme_dict.items():
                translated_higher_theme_dict[higher_keyword] = {}
                for keyword, theme_explaining in group_theme_dict.items():
                    translated_theme_explaining = APILLMParser.call_llm_api(
                        system_prompt=f"【{translated_search_words}领域专业】将用户给出的英文翻译为中文，输出仅含译文，不要含任何注释说明内容。",
                        user_prompt=theme_explaining
                    )
                    # 使用正则表达式去除括号及括号内的内容
                    translated_theme_explaining = re.sub(r'（注：[^）]*）', '', translated_theme_explaining)

                    translated_higher_theme_dict[higher_keyword][keyword] = translated_theme_explaining
            translated_higher_theme_explaining_obj[content_perspective] = translated_higher_theme_dict

        mysql_result5 = ScienceResearchMapper.update_theme_explaining({
            "research_zh_questions": str(translated_higher_theme_explaining_obj['research_questions']),
            "research_zh_methods": str(translated_higher_theme_explaining_obj['research_methods']),
            "research_zh_contributions": str(translated_higher_theme_explaining_obj['research_contributions']),
            "research_zh_limitations": str(translated_higher_theme_explaining_obj['research_limitations']),
            "task_id": self.knowledge_finding_config['task_id']
        })

        del translated_higher_theme_explaining_obj
        del higher_theme_explaining_obj

        ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'], new_message="所有主题词解释已翻译完成")

        # 翻译高表征主题词解释句子
        mysql_result6 = ScienceResearchMapper.select_theme_higher_explaining({
            "task_id": self.knowledge_finding_config['task_id']
        })

        higher_theme_explaining_obj = mysql_result6.get_data_on_results()[0]

        translated_higher_theme_explaining_obj = {}
        for content_perspective in ['research_questions', 'research_methods', 'research_contributions', 'research_limitations']:
            higher_theme_dict = eval(higher_theme_explaining_obj[content_perspective])
            translated_higher_theme_dict = {}
            for higher_keyword, theme_explaining in higher_theme_dict.items():
                translated_theme_explaining = APILLMParser.call_llm_api(
                    system_prompt=f"【{translated_search_words}领域专业】将用户给出的英文翻译为中文，输出仅含译文，不要含任何注释说明内容。",
                    user_prompt=theme_explaining
                )
                # 使用正则表达式去除括号及括号内的内容
                translated_theme_explaining = re.sub(r'（注：[^）]*）', '', translated_theme_explaining)

                translated_higher_theme_dict[higher_keyword] = translated_theme_explaining
            translated_higher_theme_explaining_obj[content_perspective] = translated_higher_theme_dict

        mysql_result7 = ScienceResearchMapper.update_theme_higher_explaining({
            "research_zh_questions": str(translated_higher_theme_explaining_obj['research_questions']),
            "research_zh_methods": str(translated_higher_theme_explaining_obj['research_methods']),
            "research_zh_contributions": str(translated_higher_theme_explaining_obj['research_contributions']),
            "research_zh_limitations": str(translated_higher_theme_explaining_obj['research_limitations']),
            "task_id": self.knowledge_finding_config['task_id']
        })

        ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'], new_message="所有高表征主题词解释已翻译完成")

    def draw_time_evolve_wordcloud(self):

        ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'], new_message="绘制时间演化领域趋势图开始")

        for content_perspective in ['research_questions', 'research_methods', 'research_contributions', 'research_limitations']:

            neo4j_result = ScienceResearchMapper.match_higher_keywords({
                "task_id": self.knowledge_finding_config['task_id']
            }, content_perspective=content_perspective)

            higher_theme_name_list = [record['name'] for record in neo4j_result.get_data_on_results()]

            for higher_theme_name in higher_theme_name_list:

                neo4j_result = ScienceResearchMapper.match_concrete_level_concrete_year_science_theme_node({
                    "task_id": self.knowledge_finding_config['task_id'],
                    "higher_theme_name": higher_theme_name
                }, content_perspective=content_perspective)

                theme_list = neo4j_result.get_data_on_results()

                mysql_result = ScienceResearchMapper.select_paper_search_where_user_id_and_task_id({
                    "user_id": self.knowledge_finding_config['user_id'],
                    "task_id": self.knowledge_finding_config['task_id']
                })

                lib_paper_list = mysql_result.get_data_on_results()

                theme_name_list = [theme_obj['theme'] for theme_obj in theme_list]

                # 获取词频权重
                word_freq = {theme_obj['theme']: theme_obj['parent_weight'][0] * theme_obj['weight'][0] for theme_obj in theme_list}

                # 获取每个主题词的高维特征向量
                group_num = 50
                total_keyword_num = len(theme_name_list)
                send_turn = total_keyword_num // group_num
                last_keyword_num = total_keyword_num % group_num
                keyword_num_list = [group_num] * send_turn + [last_keyword_num] if last_keyword_num != 0 else [group_num] * send_turn

                embedding_theme_name_list = []
                for i, keyword_num in enumerate(keyword_num_list):
                    before_num = 0 if i == 0 else sum(keyword_num_list[:i])
                    current_keyword_list = theme_name_list[before_num: before_num + keyword_num]
                    embedding_list = APILLMParser.call_embedding_llm_api(current_keyword_list)
                    embedding_theme_name_list = embedding_theme_name_list + embedding_list

                word_vectors = dict(zip(theme_name_list, embedding_theme_name_list))

                # 一个主题词可能对应多篇论文，从而有多个年份
                word_data = []
                for theme_obj in theme_list:
                    for choose_paper in theme_obj['papers']:
                        for lib_paper in lib_paper_list:
                            if lib_paper['paper_title'] == choose_paper:
                                if lib_paper['paper_publish_time'] is None or lib_paper['paper_publish_time'] == "":
                                    break

                                year = lib_paper['paper_publish_time'].split("-")[0]
                                word_data.append({
                                    "word": theme_obj['theme'],
                                    "year": int(year),
                                    "vector": word_vectors[theme_obj['theme']],
                                    "freq": word_freq[theme_obj['theme']]
                                })
                                break

                import matplotlib
                # 指定使用AGG后端，避免GUI线程冲突
                matplotlib.use('Agg')  # 在导入pyplot之前设置
                import os
                os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
                import matplotlib.pyplot as plt
                import numpy as np
                import umap
                from matplotlib.patches import Patch
                from sklearn.preprocessing import MinMaxScaler
                from adjustText import adjust_text

                # plt.rcParams['font.sans-serif'] = ['SimHei']  # 保持中文字体设置
                plt.rcParams['axes.unicode_minus'] = False

                def semantic_coordinate_mapping(word_data):
                    if len(word_data) < 2:
                        return {}

                    n_neighbors = 5 if len(word_data) > 5 else len(word_data)

                    reducer = umap.UMAP(
                        n_components=2,
                        n_neighbors=n_neighbors,
                        min_dist=0.1,
                        metric='euclidean'
                    )

                    # 提取所有向量并降维
                    vectors = np.array([item["vector"] for item in word_data])
                    coordinates = reducer.fit_transform(vectors)

                    # 坐标归一化
                    coord_scaler = MinMaxScaler(feature_range=(0, 1))
                    coordinates = coord_scaler.fit_transform(coordinates)

                    # 添加唯一标识符
                    for idx, item in enumerate(word_data):
                        item["id"] = f"{item['word']}_{item['year']}"  # 唯一标识符
                        item["coords"] = coordinates[idx]

                    return word_data

                def visualize_semantic_cloud(word_data):
                    plt.clf()
                    plt.figure(figsize=(18, 9), dpi=300)
                    ax = plt.gca()
                    ax.set_xlim(-0.5, 1.5)
                    ax.set_ylim(-0.5, 1.5)

                    COLORS = ["#7F8C8D"] * (26 - 7) + [
                        "#E74C3C",
                        "#E67E22",
                        "#F1C40F",
                        "#16A085",
                        "#27AE60",
                        "#2980B9",
                        "#8E44AD",
                        "#2C3E50"
                    ]

                    # 参数设置
                    SIZE_FACTOR = 20
                    YEAR_COLORS = dict(zip([2000 + i for i in range(26)], COLORS))

                    # 频率归一化
                    freqs = np.array([item["freq"] for item in word_data])
                    freq_scaler = MinMaxScaler(feature_range=(0.5, 1))
                    normalized_freq = freq_scaler.fit_transform(freqs.reshape(-1, 1)).flatten()

                    texts = []
                    for idx, item in enumerate(word_data):
                        txt = ax.text(
                            item["coords"][0], item["coords"][1],
                            f"{item['word']}",  # 显示年份
                            fontsize=SIZE_FACTOR * normalized_freq[idx],
                            ha='center', va='center',
                            color=YEAR_COLORS[item["year"]],
                            fontweight='semibold',
                        )
                        texts.append(txt)

                    # 标签防撞调整
                    adjust_text(texts,
                                arrowprops=None,
                                expand_text=(1.05, 1.2),
                                expand_points=(1.1, 1.3))

                    # 创建图例
                    legend_elements = [Patch(facecolor=color, label=str(year))
                                       for year, color in YEAR_COLORS.items()]
                    ax.legend(handles=legend_elements,
                              title="year",
                              loc='center left',
                              bbox_to_anchor=(1, 0.5),
                              frameon=False)

                    # 样式优化
                    ax.set_facecolor('#F5F5F5')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_color('#808080')
                    ax.spines['bottom'].set_color('#808080')

                    plt.savefig(f'{self.knowledge_finding_config["time_evolve_image_path"]}/{content_perspective}_{higher_theme_name}.png', dpi=300)

                try:
                    enhanced_data = semantic_coordinate_mapping(word_data)
                    if enhanced_data == {}:
                        continue
                    visualize_semantic_cloud(enhanced_data)

                    # 新增内存回收代码
                    plt.close('all')  # 关闭所有Matplotlib图形对象[2,5](@ref)
                    import gc
                    gc.collect()  # 立即回收图形相关内存[4,7](@ref)
                except Exception as e:
                    continue

                # 新增批量回收代码（每处理完一个主题）
                del embedding_theme_name_list  # 显式删除大对象
                plt.close('all')  # 防止多图叠加[2](@ref)
                import gc
                gc.collect()  # 及时回收中间变量内存[6,8](@ref)

            del higher_theme_name_list

        ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'], new_message="绘制完毕")

    def summary_knowledge_graph(self):

        ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'], new_message="开始总结知识网络")

        for content_perspective in ['research_questions', 'research_methods', 'research_contributions', 'research_limitations']:

            # 获取所有主题词节点
            neo4j_result = ScienceResearchMapper.match_all_theme_node_on_root_id({
                "root_id": self.knowledge_finding_config['task_id']
            }, content_perspective)

            theme_list = neo4j_result.get_data_on_results()

            # 获取所有论文节点
            neo4j_result1 = ScienceResearchMapper.match_all_paper_node_on_root_id({
                "root_id": self.knowledge_finding_config['task_id']
            }, content_perspective)

            paper_list = neo4j_result1.get_data_on_results()

            for i in range(len(theme_list)):
                theme_list[i]['zh_label'] = ""

                # 根节点无theme_explaining字段
                if not hasattr(theme_list[i], 'theme_explaining'):
                    theme_list[i]['theme_explaining'] = ""

            for i in range(len(paper_list)):
                paper_list[i]['zh_label'] = ""

                if not hasattr(paper_list[i], 'theme_explaining'):
                    paper_list[i]['theme_explaining'] = ""

            # 获取所有边
            neo4j_result2 = ScienceResearchMapper.match_all_edge_on_root_id({
                "root_id": self.knowledge_finding_config['task_id']
            }, content_perspective)

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

            higher_keyword_para_dict = {}
            for higher_summary in graph_data['children']:
                higher_keyword = higher_summary['label'].strip(" ")
                keyword_list = list(set([x['label'].strip(" *") for x in higher_summary['children']]))
                keyword_paper_dict = {x['label'].strip(" *"): ", ".join([
                    y['label'].strip(" *") for y in x['children']
                ]) for x in higher_summary['children']}

                system_prompt = f"Please use all the user-provided keywords of the thesis to generate 1 detailed paragraph on {higher_keyword} of {graph_data['label']}.(Output only one paragraph)"
                user_prompt = ", ".join(keyword_list)

                higher_keyword_para = APILLMParser.call_llm_api(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model_type="deepseek",
                    model="deepseek-chat"
                )

                higher_keyword_para = higher_keyword_para.replace("*", "")

                higher_keyword_para_phrase_list = [x for x in re.split(r"[,.]+", higher_keyword_para) if x]

                # 获取每个主题词的高维特征向量
                embedding_keyword_list = SemanticVectorCalculation.calculate_text_group(keyword_list)

                # 获取生成文本名词分割后每个名词的高维特征向量
                embedding_higher_keyword_para_phrase_list = SemanticVectorCalculation.calculate_text_group(higher_keyword_para_phrase_list)

                for i, embedding in enumerate(embedding_keyword_list):
                    max_similarity = -1  # 余弦相似度取值范围为[-1, 1]
                    best_higher_keyword_para_phrase = ""
                    for j, characterize_embedding in enumerate(embedding_higher_keyword_para_phrase_list):
                        similarity = DistanceCalculation.calculate_cosine_similarity(embedding, characterize_embedding)
                        if similarity > max_similarity:
                            max_similarity = similarity
                            best_higher_keyword_para_phrase = higher_keyword_para_phrase_list[j]

                    higher_keyword_para = higher_keyword_para.replace(best_higher_keyword_para_phrase, f"{best_higher_keyword_para_phrase}[{[m for m in keyword_paper_dict.values()][i]}]")

                higher_keyword_para_dict[higher_keyword] = higher_keyword_para

            ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'],
                                               new_message=f"段落生成完毕")

            para_list = []
            extracted_list = []
            i = 1
            for keyword, para in higher_keyword_para_dict.items():

                # 使用正确的正则表达式匹配中括号内容，确保转义 `[` 和 `]`
                pattern = re.compile(r'\[(.*?)\]')

                def replacer(match):
                    content = match.group(1)
                    extracted_list.append(content)
                    return f'[{len(extracted_list)}]'

                # 执行替换
                para = pattern.sub(replacer, para)

                zh_keyword = APILLMParser.call_llm_api(
                    system_prompt="请你将用户给出的英文翻译为中文。（仅输出译文，不要含任何注释说明内容）",
                    user_prompt=keyword
                )

                # 使用正则表达式去除括号及括号内的内容
                zh_keyword = re.sub(r'（注：[^）]*）', '', zh_keyword)

                example_input = "Maintenance strategies and lifecycle analysis of wind turbine failure encompass a comprehensive approach integrating condition-based maintenance (CBM)[1][2][3][4][5][6], predictive maintenance[7], and reliability-based design optimization (RBDO) to enhance turbine longevity and reduce operational costs[8]. Competitive failure models with fault incubation periods and probabilistic S-N curve modeling enable accurate wear calculation and failure prediction[9][10], while digital twin technology and the StrathOW-OM operational simulation model facilitate real-time monitoring and maintenance strategy evaluation[11][12]. Wear model comparisons and incomplete maintenance parameter integration refine maintenance schedules[13][14][15][16], whereas martingale analysis and sensitivity studies assess cost factors and strategy effectiveness. Database relationships and expert-labeled work orders improve maintenance event rate calculations[17][18], and time-based incomplete maintenance (TBIM) strategies optimize interventions[19][20][21]. Interpolation methods for failure envelopes and mathematical maintenance modeling further support lifecycle analysis[22], ensuring robust, cost-efficient wind turbine operations through data-driven decision-making."
                example_output = "风力涡轮机故障的维护策略和生命周期分析涵盖了一种综合方法，该方法集成了基于状态的维护（CBM）[1][2][3][4][5][6]、预测性维护[7]和基于可靠性的设计优化（RBDO），以延长涡轮机寿命并降低运营成本[8]。具有故障潜伏期的竞争性故障模型和概率S-N曲线建模能够实现精确的磨损计算和故障预测[9][10]，而数字孪生技术和StrathOW-OM运营仿真模型则有助于实时监控和维护策略评估[11][12]。磨损模型比较和不完整维护参数整合优化了维护计划[13][14][15][16]，而马丁格尔分析和敏感性研究则评估了成本因素和策略有效性。数据库关系和专家标记的工作订单改进了维护事件率计算[17][18]，而基于时间的不完整维护（TBIM）策略优化了干预措施[19][20][21]。故障包络线的插值方法和数学维护建模进一步支持了生命周期分析[22]，通过数据驱动的决策确保了风力涡轮机运营的稳健性和成本效益。"
                max_retries = 5
                zh_para = ""
                for _ in range(max_retries):
                    zh_para = APILLMParser.call_llm_api(
                        system_prompt=f"请你将用户给出的英文翻译为中文。（仅输出译文，不要含任何注释说明内容）【示例输入：{example_input}，示例输出：{example_output}】",
                        user_prompt=para
                    )
                    # 使用正则表达式去除括号及括号内的内容
                    zh_para = re.sub(r'（注：[^）]*）', '', zh_para)

                    if "json" not in zh_para and r"\times" not in zh_para and "\times" not in zh_para:
                        break

                zh_para = zh_para.replace("*", "")

                para_list.append(f'---  \n### {i}.{keyword}  \n### {i}.{zh_keyword}  \n- {para}  \n\n- {zh_para}')

                i += 1

            title = graph_data['label']
            zh_title = APILLMParser.call_llm_api(
                system_prompt="请你将用户给出的英文翻译为中文。（仅输出译文，不要含任何注释说明内容）",
                user_prompt=title
            )

            # 使用正则表达式去除括号及括号内的内容
            zh_title = re.sub(r'（注：[^）]*）', '', zh_title)

            paragraph_text = "  \n".join(para_list)

            ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'],
                                               new_message=f"段落翻译完毕")

            content_perspective_mapping = {
                "research_questions": "研究问题",
                "research_methods": "研究方法",
                "research_contributions": "研究贡献",
                "research_limitations": "研究局限"
            }

            paragraph = f"# {title} ({content_perspective})  \n# {zh_title} ({content_perspective_mapping[content_perspective]})  \n\n{paragraph_text} \n\n# 参考文献"

            j = 1
            for paper_name in extracted_list:
                paragraph += f"  \n[{j}]  {paper_name}"
                j += 1

            report_id = RandomStrGenerator.generate_uuid()

            ScienceResearchMapper.insert_graph({
                "report_id": report_id,
                "task_id": self.knowledge_finding_config['task_id'],
                "content_perspective": content_perspective,
                "paragraph": paragraph,
            })

            del theme_list
            del graph_data

        ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'], new_message="总结完毕")

    def reconstruct_graph(self):

        ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'], new_message="开始重构知识网络")

        mysql_result = ScienceResearchMapper.select_theme_filtered_where_user_id_and_task_id({
            "user_id": self.knowledge_finding_config['user_id'],
            "task_id": self.knowledge_finding_config['task_id'],
        })

        theme_filtered = mysql_result.get_data_on_results()[0]

        mysql_result02 = ScienceResearchMapper.select_theme_explaining({
            "task_id": self.knowledge_finding_config['task_id'],
        })
        paper_summary_explaining = mysql_result02.get_data_on_results()[0]

        # 创建新的桥接主题词节点
        for content_perspective in ['research_questions', 'research_methods', 'research_contributions', 'research_limitations']:
            filtered_str = theme_filtered[content_perspective]
            try:
                filtered_dict = eval(filtered_str)
                perspective_paper_summary_explaining = eval(paper_summary_explaining[content_perspective])
            except Exception as e:
                print(e)
                continue
            for characterize_theme, theme_dict in filtered_dict.items():
                for theme, filtered_theme in theme_dict.items():

                    # 获取桥接节点+边数据
                    neo4j_result = ScienceResearchMapper.new_bridge_to_get_old_data({
                        "user_id": self.knowledge_finding_config['user_id'],
                        "task_id": self.knowledge_finding_config['task_id'],
                        "higher_theme_name": characterize_theme,
                        "theme_name": theme
                    }, content_perspective)

                    # 删除旧桥接节点
                    neo4j_result1 = ScienceResearchMapper.new_bridge_to_delete_old_data({
                        "user_id": self.knowledge_finding_config['user_id'],
                        "task_id": self.knowledge_finding_config['task_id'],
                        "higher_theme_name": characterize_theme,
                        "theme_name": theme
                    }, content_perspective)

                    bridge_id = RandomStrGenerator.generate_uuid()

                    # 创建新桥接节点
                    for new_record in neo4j_result.get_data_on_results():

                        neo4j_result2 = ScienceResearchMapper.new_bridge_to_create_new_data({
                            "user_id": self.knowledge_finding_config['user_id'],
                            "task_id": self.knowledge_finding_config['task_id'],
                            "higher_theme_name": characterize_theme,
                            "paper_id": new_record['paper_id'],
                            "left_weight": new_record['left_weight'],
                            "bridge_name": filtered_theme,
                            "right_weight": new_record['right_weight'],
                            "bridge_id": bridge_id,
                            "theme_explaining": perspective_paper_summary_explaining[characterize_theme][filtered_theme]
                        }, content_perspective)

            del filtered_str

        ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'], new_message="重构完成")

    def create_graph(self):

        ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'], new_message="开始生成图结构")

        # 创建检索主题词节点
        neo4j_result = ScienceResearchMapper.merge_science_theme_node_on_user_node_to_search({
            "science_theme_name": self.knowledge_finding_config['search_words'],
            "user_id": self.knowledge_finding_config['user_id'],
            "science_theme_id": self.knowledge_finding_config['task_id']
        })

        mysql_result = ScienceResearchMapper.select_theme_characterize_where_user_id_and_task_id({
            "user_id": self.knowledge_finding_config['user_id'],
            "task_id": self.knowledge_finding_config['task_id'],
        })
        paper_higher_summary = mysql_result.get_data_on_results()[0]

        mysql_result01 = ScienceResearchMapper.select_theme_higher_explaining({
            "task_id": self.knowledge_finding_config['task_id'],
        })
        paper_higher_summary_explaining = mysql_result01.get_data_on_results()[0]

        mysql_result02 = ScienceResearchMapper.select_theme_explaining({
            "task_id": self.knowledge_finding_config['task_id'],
        })
        paper_summary_explaining = mysql_result02.get_data_on_results()[0]

        # 从四个角度构建主题树
        for content_perspective in ['research_questions', 'research_methods', 'research_contributions',
                                    'research_limitations']:

            try:
                summary_relations = eval(paper_higher_summary[content_perspective])
                perspective_paper_higher_summary_explaining = eval(paper_higher_summary_explaining[content_perspective])
                perspective_paper_summary_explaining = eval(paper_summary_explaining[content_perspective])
            except Exception as e:
                ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'], new_message=f"paper_higher_summary: {content_perspective} 读取出错: {e}")
                continue

            total_higher_theme_num = sum([len(x) for x in summary_relations.values()])

            for higher_keyword, summary_dict in summary_relations.items():

                # 创建高表征主题词节点
                higher_theme_id = RandomStrGenerator.generate_uuid()
                neo4j_result1 = ScienceResearchMapper.merge_science_higher_theme({
                    "user_id": self.knowledge_finding_config['user_id'],
                    "task_id": self.knowledge_finding_config['task_id'],
                    "edge_weight": round(float(len(summary_dict)) / total_higher_theme_num, 2),
                    "theme_id": higher_theme_id,
                    "theme_name": higher_keyword,
                    "theme_explaining": perspective_paper_higher_summary_explaining[higher_keyword]
                }, content_perspective)

                for keyword, edge_weight in summary_dict.items():

                    # 创建主题词节点
                    theme_id = RandomStrGenerator.generate_uuid()
                    neo4j_result2 = ScienceResearchMapper.merge_theme({
                        "user_id": self.knowledge_finding_config['user_id'],
                        "task_id": self.knowledge_finding_config['task_id'],
                        "higher_theme_id": higher_theme_id,
                        "edge_weight": edge_weight,
                        "theme_id": theme_id,
                        "theme_name": keyword,
                        "theme_explaining": perspective_paper_summary_explaining[higher_keyword][keyword]
                    }, content_perspective)

                    mysql_result4 = ScienceResearchMapper.select_paper_search_where_user_id_and_task_id({
                        "user_id": self.knowledge_finding_config['user_id'],
                        "task_id": self.knowledge_finding_config['task_id'],
                    })

                    paper_summary_list = mysql_result4.get_data_on_results()

                    for paper_summary in paper_summary_list:

                        # 跳过没有原文pdf的论文
                        if paper_summary[content_perspective] is None:
                            continue

                        try:
                            paper_summary_perspective_dict = eval(paper_summary[content_perspective])
                        except Exception as e:
                            ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'],
                                                               new_message=f"paper_summary: {content_perspective} 出错")
                            continue

                        if keyword in [x for x in paper_summary_perspective_dict.keys()]:
                            # 创建论文节点
                            paper_id = paper_summary['paper_id']
                            paper_title = paper_summary['paper_title']

                            neo4j_result3 = ScienceResearchMapper.merge_science_paper_node({
                                "theme_id": theme_id,
                                "paper_id": paper_id,
                                "paper_name": paper_title,
                                "edge_weight": paper_summary_perspective_dict[keyword]
                            }, content_perspective)

            del summary_relations

        ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'], new_message="构建完毕")

    def characterize_themes(self):

        ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'], new_message="开始生成高表征关键词")

        mysql_result = ScienceResearchMapper.select_theme_characterize_where_user_id_and_task_id({
            "user_id": self.knowledge_finding_config['user_id'],
            "task_id": self.knowledge_finding_config['task_id'],
        })

        if mysql_result.verify_data_on_results():
            mysql_result1 = ScienceResearchMapper.delete_theme_characterize({
                "task_id": self.knowledge_finding_config['task_id']
            })

        mysql_result2 = ScienceResearchMapper.select_paper_search_where_user_id_and_task_id({
            "user_id": self.knowledge_finding_config['user_id'],
            "task_id": self.knowledge_finding_config['task_id']
        })

        paper_summary_list = mysql_result2.get_data_on_results()

        total_theme_dict = {}
        total_characterize_filtered_dict = {}
        total_theme_explaining_dict = {}
        total_new_theme_dict = {}
        total_characterize_theme_dict = {}

        for content_perspective in ['research_questions', 'research_methods', 'research_contributions', 'research_limitations']:
            content_perspective_keyword_list = []
            content_perspective_explaining_list = []
            content_perspective_total_list = []
            for paper_summary in paper_summary_list:
                # 跳过没有原文pdf的论文
                if paper_summary[content_perspective] is not None:
                    content_perspective_keyword_list = content_perspective_keyword_list + [x for x in eval(paper_summary[content_perspective]).keys()]
                    content_perspective_explaining_list = content_perspective_explaining_list + [x for x in eval(paper_summary[f"{content_perspective}_explaining"]).values()]
                    content_perspective_total_list.append(paper_summary[f"{content_perspective}_explaining"])

            # 生成一段描述

            system_prompt = f"Please generate a paragraph based on the data provided by the user about {content_perspective} of {self.knowledge_finding_config['search_words']}."
            user_prompt = ", ".join(content_perspective_keyword_list)

            combined_explaining_text = APILLMParser.call_llm_api(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_type="deepseek",
                model="deepseek-chat"
            )

            example_output = """
                {
                    'Failure Rate Sensitivity': 'The text emphasizes how failure definitions, data selection, downtime limits, and repair constraints influence failure rate calculations. These factors are critical for understanding the sensitivity and variability of failure rates in reliability studies.',
                    'Standardization of Failure Definitions': 'Standardizing failure definitions is highlighted as a key challenge to reduce uncertainty in reliability analysis. Inconsistent definitions lead to ambiguous data, complicating cross-study comparisons and industry-wide reliability assessments.',
                    'Fault Detection Techniques': 'Advanced methods like SCADA data anomaly detection using unsupervised learning and domain-specific knowledge are explored to improve early warning systems. These techniques aim to enhance predictive accuracy and reduce undetected faults.',
                    'Structural Integrity of FOWTs': 'Research on floating offshore wind turbines (FOWTs) focuses on tendon and mooring system failures, wind-wave coupling effects, and dynamic responses under extreme conditions. These studies address unique structural challenges in offshore environments.',
                    'Predictive Maintenance Strategies': 'Digital twin frameworks and deep learning are investigated for gearbox and bearing fault diagnosis. These strategies aim to transition from reactive to proactive maintenance, minimizing downtime and operational costs.',
                    'Material Degradation Analysis': 'Studies on turbine blade material degradation and manufacturing defects seek to extend service life. Understanding these factors helps mitigate premature failures and improve component durability.',
                    'Economic and Operational Challenges': 'The text identifies challenges in multi-rotor systems, life-cycle cost optimization, and scalability. Addressing these issues is crucial for improving cost-efficiency and operational feasibility in wind energy projects.',
                    'AI and Big Data Integration': 'Integration of AI and big data in wind farm maintenance is explored to optimize decision-making. These technologies enable real-time analytics and adaptive maintenance planning, enhancing overall system reliability.'
                }
              """

            characterize_theme_dict = {}
            max_retries = 5
            for _ in range(max_retries):

                system_prompt = f"Please summarize the text provided by the user into high-level representative keywords about {content_perspective} of {self.knowledge_finding_config['search_words']}, and provide a short paragraph of reasoning of each keyword to the text in English. (all of them in JSON format) {{Example Output: {example_output}}}"
                user_prompt = combined_explaining_text

                characterize_theme_dict = APILLMParser.call_llm_api(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    resp_format="json"
                )

                if characterize_theme_dict != {} and isinstance(characterize_theme_dict, dict):
                    break

            characterize_theme_list = [x for x in characterize_theme_dict.keys()]
            total_characterize_theme_dict[content_perspective] = characterize_theme_dict

            # 获取每个主题词的高维特征向量
            total_keyword_embedding_list = SemanticVectorCalculation.calculate_text_group(content_perspective_keyword_list)

            # 获取每个高表征主题词的高维特征向量
            total_characterize_embedding_list = SemanticVectorCalculation.calculate_text_group(characterize_theme_list)

            # 计算余弦相似度
            theme_dict = {characterize_theme: {} for characterize_theme in characterize_theme_list}
            theme_explaining_dict = {characterize_theme: {} for characterize_theme in characterize_theme_list}
            theme_embedding_dict = {characterize_theme: {} for characterize_theme in characterize_theme_list}
            for i, embedding in enumerate(total_keyword_embedding_list):
                max_similarity = -1  # 余弦相似度取值范围为[-1, 1]
                best_characterize_theme = ""
                for j, characterize_embedding in enumerate(total_characterize_embedding_list):
                    similarity = DistanceCalculation.calculate_cosine_similarity(embedding, characterize_embedding)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_characterize_theme = characterize_theme_list[j]

                if best_characterize_theme in theme_dict.keys():
                    # 权重分数保留两位小数
                    score = round((max_similarity + 1) / 2.0, 2)
                    theme_dict[best_characterize_theme][content_perspective_keyword_list[i]] = score
                    theme_explaining_dict[best_characterize_theme][content_perspective_keyword_list[i]] = content_perspective_explaining_list[i]
                    theme_embedding_dict[best_characterize_theme][content_perspective_keyword_list[i]] = embedding

            # 删除没有子主题词的高表征主题词
            valid_theme_dict = {}
            valid_theme_explaining_dict = {}
            valid_theme_embedding_dict = {}
            for keyword, children_dict in theme_dict.items():
                if len(children_dict) > 0:
                    valid_theme_dict[keyword] = children_dict
                    valid_theme_explaining_dict[keyword] = theme_explaining_dict[keyword]
                    valid_theme_embedding_dict[keyword] = theme_embedding_dict[keyword]

            theme_dict = valid_theme_dict
            theme_explaining_dict = valid_theme_explaining_dict
            theme_embedding_dict = valid_theme_embedding_dict

            total_theme_dict[content_perspective] = theme_dict
            total_theme_explaining_dict[content_perspective] = theme_explaining_dict

            ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'], new_message=f"{content_perspective} 已生成")

        characterize_id = RandomStrGenerator.generate_uuid()

        mysql_result3 = ScienceResearchMapper.insert_theme_characterize({
            "characterize_id": characterize_id,
            "user_id": self.knowledge_finding_config['user_id'],
            "task_id": self.knowledge_finding_config['task_id'],
            "research_questions": str(total_theme_dict['research_questions']),
            "research_methods": str(total_theme_dict['research_methods']),
            "research_contributions": str(total_theme_dict['research_contributions']),
            "research_limitations": str(total_theme_dict['research_limitations']),
        })

        explaining_id = RandomStrGenerator.generate_uuid()

        mysql_result5 = ScienceResearchMapper.insert_theme_higher_explaining({
            "explaining_id": explaining_id,
            "user_id": self.knowledge_finding_config['user_id'],
            "task_id": self.knowledge_finding_config['task_id'],
            "research_questions": str(total_characterize_theme_dict['research_questions']),
            "research_methods": str(total_characterize_theme_dict['research_methods']),
            "research_contributions": str(total_characterize_theme_dict['research_contributions']),
            "research_limitations": str(total_characterize_theme_dict['research_limitations']),
        })

        explaining_id1 = RandomStrGenerator.generate_uuid()

        mysql_result5 = ScienceResearchMapper.insert_theme_explaining({
            "explaining_id": explaining_id1,
            "user_id": self.knowledge_finding_config['user_id'],
            "task_id": self.knowledge_finding_config['task_id'],
            "research_questions": str(total_theme_explaining_dict['research_questions']),
            "research_methods": str(total_theme_explaining_dict['research_methods']),
            "research_contributions": str(total_theme_explaining_dict['research_contributions']),
            "research_limitations": str(total_theme_explaining_dict['research_limitations']),
        })

        ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'], new_message="生成完毕")

            # # 重分组+重定位
            # new_theme_dict = {characterize_theme: {} for characterize_theme in characterize_theme_list}
            # new_theme_explaining_dict = {characterize_theme: {} for characterize_theme in characterize_theme_list}
            # new_theme_embedding_dict = {characterize_theme: {} for characterize_theme in characterize_theme_list}
            # characterize_filtered_dict = {}
            # for characterize_theme, group_themes in theme_embedding_dict.items():
            #
            #     time.sleep(2)
            #
            #     group_themes_list = [x for x in group_themes.keys()]
            #
            #     group_themes_embeddings = theme_explaining_dict[characterize_theme]
            #
            #     # 生成一段描述
            #
            #     system_prompt = f"Please generate a detailed paragraph based on the data provided by the user about {content_perspective} of {characterize_theme} in {self.knowledge_finding_config['search_words']}."
            #     user_prompt = str(group_themes_embeddings)
            #     small_combined_explaining_text = APILLMParser.call_llm_api(
            #         system_prompt=system_prompt,
            #         user_prompt=user_prompt,
            #         model_type="deepseek",
            #         model="deepseek-chat"
            #     )
            #
            #     example_output = """
            #         {
            #             'Failure Definitions and Data Selection Criteria': 'The study emphasizes how varying definitions of failure (e.g., minor vs. catastrophic) and criteria for selecting operational data (e.g., filtering thresholds, timeframes) directly influence the calculated failure rates. These variations affect the reliability of comparisons between wind turbine systems and necessitate standardized frameworks for accurate analysis.',
            #             'I-SAE Method for Anomaly Detection': 'The proposed I-SAE method dynamically filters training data to amplify reconstruction error differences between normal and abnormal states, improving anomaly detection accuracy. This addresses limitations in traditional methods that struggle with overlapping operational data distributions, enhancing failure prediction capabilities.',
            #             'Multi-Rotor Wind Turbine Systems (MRS) Impact on Operational Expenditure': 'The research analyzes how failure rates in MRS configurations affect operational costs compared to single-rotor turbines. Despite potential redundancy benefits, higher failure rates in MRS may negate cost savings, highlighting the need for robust reliability engineering to optimize maintenance strategies.',
            #             'Financial Competitiveness of MRS through Failure Rate Reduction': 'The study quantifies the required reduction in MRS failure rates to achieve cost parity with conventional wind farms. This underscores the economic imperative to improve component durability and system design, ensuring MRS viability in competitive energy markets.',
            #             'Fatigue-Related Failure Scaling in MRS Configurations': 'The text examines how fatigue strength and component size scaling in MRS affect failure rates. Smaller components in multi-rotor systems may experience accelerated fatigue degradation due to stress concentrations, necessitating material innovations and design adjustments to mitigate failure risks.',
            #             'Role of Failure Inspection in Structural Integrity': 'The research stresses the importance of systematic failure inspections to detect early signs of wear, corrosion, or structural flaws. Effective inspection protocols are critical for preventing cascading failures, ensuring compliance with safety standards, and extending turbine lifespan.'
            #         }
            #       """
            #
            #     new_generated_keyword_theme_dict = {}
            #     max_retries = 5
            #     for _ in range(max_retries):
            #
            #         system_prompt = f"Please summarize the text provided by the user into at least 6 concrete keywords about {content_perspective} of {characterize_theme} in {self.knowledge_finding_config['search_words']}, and provide a short paragraph of reasoning of each keyword to the text in English. (all of them in JSON format) {{Example Output: {example_output}}}"
            #         user_prompt = small_combined_explaining_text
            #
            #         new_generated_keyword_theme_dict = APILLMParser.call_llm_api(
            #             system_prompt=system_prompt,
            #             user_prompt=user_prompt,
            #             resp_format="json"
            #         )
            #
            #         if new_generated_keyword_theme_dict != {} and isinstance(new_generated_keyword_theme_dict, dict):
            #             break
            #
            #     new_generated_keyword_theme_list = [x for x in new_generated_keyword_theme_dict.keys()]
            #     new_generated_keyword_theme_explaining_list = [x for x in new_generated_keyword_theme_dict.values()]
            #
            #     # 过滤掉科研通用词：【literature review】
            #     new_generated_keyword_theme_list = [x for x in new_generated_keyword_theme_list if "literature review" not in x.lower() and "review of literature" not in x.lower()]
            #
            #     if len(group_themes_list) == 0:
            #         ScienceResearchProgress.update_log(self.knowledge_finding_config['user_id'], f"报错: {characterize_theme}: {group_themes_list}")
            #         continue
            #
            #     ScienceResearchProgress.update_log(self.knowledge_finding_config['user_id'], f"修正后比例: {characterize_theme}: {round(len(new_generated_keyword_theme_list) / len(group_themes_list), 2)}")
            #
            #     characterize_theme_embedding = SemanticVectorCalculation.calculate_text_group([characterize_theme])[0]
            #
            #     # 获取每个新的主题词的高维特征向量
            #     new_generated_keyword_embedding_list = SemanticVectorCalculation.calculate_text_group(new_generated_keyword_theme_list)
            #
            #     # 计算新的主题词与高表征主题词的余弦相似度
            #     for i, embedding in enumerate(new_generated_keyword_embedding_list):
            #         max_similarity = DistanceCalculation.calculate_cosine_similarity(embedding, characterize_theme_embedding)
            #         # 权重分数保留两位小数
            #         score = round((max_similarity + 1) / 2.0, 2)
            #         new_theme_dict[characterize_theme][new_generated_keyword_theme_list[i]] = score
            #         new_theme_explaining_dict[characterize_theme][new_generated_keyword_theme_list[i]] = new_generated_keyword_theme_explaining_list[i]
            #         new_theme_embedding_dict[characterize_theme][new_generated_keyword_theme_list[i]] = embedding
            #
            #     # 计算原有主题词与新的主题词的余弦相似度
            #     filtered_dict = {}
            #     for i, embedding in enumerate(group_themes.values()):
            #         max_similarity = -1  # 余弦相似度取值范围为[-1, 1]
            #         best_filtered_theme = ""
            #         for j, filtered_embedding in enumerate(new_generated_keyword_embedding_list):
            #             similarity = DistanceCalculation.calculate_cosine_similarity(embedding, filtered_embedding)
            #             if similarity > max_similarity:
            #                 max_similarity = similarity
            #                 best_filtered_theme = new_generated_keyword_theme_list[j]
            #
            #         filtered_dict[group_themes_list[i]] = best_filtered_theme
            #
            #     characterize_filtered_dict[characterize_theme] = filtered_dict
            #
            # total_characterize_filtered_dict[content_perspective] = characterize_filtered_dict
            # total_theme_explaining_dict[content_perspective] = new_theme_explaining_dict
            #
            # content_perspective_new_theme_dict = {}
            # for higher_keyword, old_to_new_keyword_dict in characterize_filtered_dict.items():
            #     content_perspective_new_theme_dict[higher_keyword] = {old_keyword: new_theme_dict[higher_keyword][new_keyword] for old_keyword, new_keyword in old_to_new_keyword_dict.items()}
            #
            # total_new_theme_dict[content_perspective] = content_perspective_new_theme_dict
            #
            # del content_perspective_keyword_list
            # del new_generated_keyword_theme_dict
            # del small_combined_explaining_text
        #
        # characterize_id = RandomStrGenerator.generate_uuid()
        #
        # # 主题词是旧的，主题词对应的分数是新主题词的分数: total_new_theme_dict
        # mysql_result3 = ScienceResearchMapper.insert_theme_characterize({
        #     "characterize_id": characterize_id,
        #     "user_id": self.knowledge_finding_config['user_id'],
        #     "task_id": self.knowledge_finding_config['task_id'],
        #     "research_questions": str(total_new_theme_dict['research_questions']),
        #     "research_methods": str(total_new_theme_dict['research_methods']),
        #     "research_contributions": str(total_new_theme_dict['research_contributions']),
        #     "research_limitations": str(total_new_theme_dict['research_limitations']),
        # })
        #
        # filtered_id = RandomStrGenerator.generate_uuid()
        #
        # # 每个旧的主题词对应新的主题词
        # mysql_result4 = ScienceResearchMapper.insert_theme_filtered({
        #     "filtered_id": filtered_id,
        #     "user_id": self.knowledge_finding_config['user_id'],
        #     "task_id": self.knowledge_finding_config['task_id'],
        #     "research_questions": str(total_characterize_filtered_dict['research_questions']),
        #     "research_methods": str(total_characterize_filtered_dict['research_methods']),
        #     "research_contributions": str(total_characterize_filtered_dict['research_contributions']),
        #     "research_limitations": str(total_characterize_filtered_dict['research_limitations']),
        # })
        #
        # explaining_id = RandomStrGenerator.generate_uuid()
        #
        # mysql_result5 = ScienceResearchMapper.insert_theme_higher_explaining({
        #     "explaining_id": explaining_id,
        #     "user_id": self.knowledge_finding_config['user_id'],
        #     "task_id": self.knowledge_finding_config['task_id'],
        #     "research_questions": str(total_characterize_theme_dict['research_questions']),
        #     "research_methods": str(total_characterize_theme_dict['research_methods']),
        #     "research_contributions": str(total_characterize_theme_dict['research_contributions']),
        #     "research_limitations": str(total_characterize_theme_dict['research_limitations']),
        # })
        #
        # explaining_id1 = RandomStrGenerator.generate_uuid()
        #
        # mysql_result5 = ScienceResearchMapper.insert_theme_explaining({
        #     "explaining_id": explaining_id1,
        #     "user_id": self.knowledge_finding_config['user_id'],
        #     "task_id": self.knowledge_finding_config['task_id'],
        #     "research_questions": str(total_theme_explaining_dict['research_questions']),
        #     "research_methods": str(total_theme_explaining_dict['research_methods']),
        #     "research_contributions": str(total_theme_explaining_dict['research_contributions']),
        #     "research_limitations": str(total_theme_explaining_dict['research_limitations']),
        # })
        #
        # ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'], new_message="生成完毕")

    def characterize_themes_cluster_method(self):

        ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'], new_message="开始生成高表征关键词")

        mysql_result = ScienceResearchMapper.select_theme_characterize_where_user_id_and_task_id({
            "user_id": self.knowledge_finding_config['user_id'],
            "task_id": self.knowledge_finding_config['task_id'],
        })

        if mysql_result.verify_data_on_results():
            mysql_result1 = ScienceResearchMapper.delete_theme_characterize({
                "task_id": self.knowledge_finding_config['task_id']
            })

        mysql_result2 = ScienceResearchMapper.select_paper_search_where_user_id_and_task_id({
            "user_id": self.knowledge_finding_config['user_id'],
            "task_id": self.knowledge_finding_config['task_id']
        })

        paper_summary_list = mysql_result2.get_data_on_results()

        total_theme_dict = {}
        for content_perspective in ['research_questions', 'research_methods', 'research_contributions', 'research_limitations']:
            content_perspective_list = []
            for paper_summary in paper_summary_list:
                # 跳过没有原文pdf的论文
                if paper_summary[content_perspective] is not None:
                    content_perspective_list = content_perspective_list + [x for x in eval(paper_summary[content_perspective]).keys()]

            # 主题词去重
            content_perspective_list = list(set(content_perspective_list))

            # 获取每个主题词的高维特征向量
            group_num = 50
            total_keyword_num = len(content_perspective_list)
            send_turn = total_keyword_num // group_num
            last_keyword_num = total_keyword_num % group_num
            keyword_num_list = [group_num] * send_turn + [last_keyword_num]

            total_embedding_list = []
            for i, keyword_num in enumerate(keyword_num_list):
                before_num = 0 if i == 0 else sum(keyword_num_list[:i])
                current_keyword_list = content_perspective_list[before_num: before_num + keyword_num]
                embedding_list = APILLMParser.call_embedding_llm_api(current_keyword_list)
                total_embedding_list = total_embedding_list + embedding_list

            # 计算相似度矩阵
            similarity_matrix = DistanceCalculation.calculate_cosine_similarity(total_embedding_list)

            # 主题词聚类
            labels, cluster_num = ClusterClassification.hierarchical_clustering(similarity_matrix)

            cluster_dict = {}
            for i, label in enumerate(labels):
                if label not in cluster_dict.keys():
                    cluster_dict[label] = []
                cluster_dict[label].append(content_perspective_list[i])

            cluster_list = [{
                "id": m,
                "data": cluster
            } for m, cluster in enumerate(cluster_dict.values())]

            example_input = """
                [
                    {   
                        'id': 0,
                        'data': [
                            'CEEMDAN (Complete Ensemble Empirical Mode Decomposition with Adaptive Noise)',
                            'Improved TFR (Time-Frequency Representation) demodulation',
                            'CWT (Continuous Wavelet Transform)',
                            'EMD (Empirical Mode Decomposition) analysis',
                            'Wavelet transform for time-frequency analysis',
                            'Resonance demodulation comparison',
                            'Signal processing techniques for fault detection'
                        ]
                    },
                    {
                        'id': 1,
                        'data': [
                            'Page-Hinkley (P-H) statistical test for fault detection',
                            'Monte Carlo simulation for lifecycle maintenance',
                            'Statistical waiting time tool for downtime prediction',
                            'KC (Kurtosis-Correlation) indicator',
                            'Sensitivity analysis for failure rates',
                            'Failure rate classification (global/local)',
                            'Quantitative risk assessment in wind energy',
                            'Sensitivity analysis for wind resource assessment'
                        ]
                    },
                    {
                        'id': 2,
                        'data': [
                            'Simulation and experimental validation (IMS/CWRU datasets)',
                            'Experimental approach for FOWT analysis',
                            'Free-decay tests for resonance properties',
                            'Regular wave tests for RAOs',
                            'Wind-wave combined tests for dynamic performance',
                            'Mooring system simulation for tension force analysis',
                            'Froude scaling for model testing',
                            'Optical tracking system for motion capture'
                        ]
                    },
                    {
                        'id': 3,
                        'data': [
                            'Extended Kalman Filter (EKF) bank structured according to GOS',
                            'Nonlinear model for RSC and linear model for GSC',
                            'Bank of Luenberger observers for fault diagnosis',
                            'Sliding mode observers for fault detection',
                            'Clarke transformation for reference frame conversion',
                            'Jacobian's discrete-time method for nonlinearity',
                            'Logical combination for fault localization',
                        ]
                    },
                    {
                        'id': 4,
                        'data': [
                            'Machine learning algorithms for fault diagnosis',
                            'Deep learning models for wind speed forecasting',
                            'Artificial neural networks for O&M optimization',
                            'Data collection and analysis from SCADA systems',
                            'IoT-based platforms for wind turbine monitoring',
                            'Digital twin technology for wind turbine simulation'
                        ]
                    },
                    {
                        'id': 5,
                        'data': [
                            'Life-cycle cost analysis for wind energy projects',
                            'Cost-based component comparison parameter',
                            'Decision support systems for O&M decision-making',
                            'Fatigue strength scaling theory application'
                        ]
                    }
                ]
            """

            example_output = """
                [
                    {
                        'id': 0,
                        'keyword': 'Signal Decomposition and Time-Frequency Analysis Methods'
                    },
                    {
                        'id': 1,
                        'keyword': 'Statistical Analysis and Fault Detection Metrics'
                    },
                    {
                        'id': 2,
                        'keyword': 'Experimental and Simulation-Based Validation Techniques'
                    },
                    {
                        'id': 3,
                        'keyword': 'Modeling and Observer-Based Diagnostic Methods'
                    },
                    {
                        'id': 4,
                        'keyword': 'Data-Driven and Machine Learning Approaches'
                    },
                    {
                        'id': 5,
                        'keyword': 'Cost and Lifecycle Analysis Methods'
                    }
                ]
            """

            system_prompt = f"Please generate high-representative keyword summary for each cluster provided by the user on {content_perspective.replace('_', ' ')} of academic papers according to the id. (all of them in JSON format) {{Example Input: {example_input}}} {{Example Output: {example_output}}}"
            user_prompt = str(cluster_list)

            characterize_theme_cluster_list = APILLMParser.call_llm_api(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                resp_format="json",
                model_type="deepseek",
                model="deepseek-chat"
            )

            # 转换为字典
            dict_a = {item['id']: item['data'] for item in cluster_list}
            dict_b = {item['id']: item['keyword'] for item in characterize_theme_cluster_list}

            # 对齐并生成目标字典
            theme_dict = {dict_b[id]: dict_a[id] for id in dict_a if id in dict_b}

            total_theme_dict[content_perspective] = theme_dict

            ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'],
                                               new_message=f"{content_perspective} 已生成")

        characterize_id = RandomStrGenerator.generate_uuid()

        mysql_result3 = ScienceResearchMapper.insert_theme_characterize({
            "characterize_id": characterize_id,
            "user_id": self.knowledge_finding_config['user_id'],
            "task_id": self.knowledge_finding_config['task_id'],
            "research_questions": str(total_theme_dict['research_questions']),
            "research_methods": str(total_theme_dict['research_methods']),
            "research_contributions": str(total_theme_dict['research_contributions']),
            "research_limitations": str(total_theme_dict['research_limitations'])
        })

        ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'], new_message="生成完毕")

    def characterize_themes_direct_llm_method(self):

        ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'], new_message="开始生成高表征关键词")

        mysql_result = ScienceResearchMapper.select_theme_characterize_where_user_id_and_task_id({
            "user_id": self.knowledge_finding_config['user_id'],
            "task_id": self.knowledge_finding_config['task_id'],
        })

        if mysql_result.verify_data_on_results():
            mysql_result1 = ScienceResearchMapper.delete_theme_characterize({
                "task_id": self.knowledge_finding_config['task_id']
            })

        mysql_result2 = ScienceResearchMapper.select_paper_search_where_user_id_and_task_id({
            "user_id": self.knowledge_finding_config['user_id'],
            "task_id": self.knowledge_finding_config['task_id']
        })

        paper_summary_list = mysql_result2.get_data_on_results()

        total_theme_dict = {}
        for content_perspective in ['research_questions', 'research_methods', 'research_contributions', 'research_limitations']:
            content_perspective_list = []
            for paper_summary in paper_summary_list:
                # 跳过没有原文pdf的论文
                if paper_summary[content_perspective] is not None:
                    content_perspective_list = content_perspective_list + [x for x in eval(paper_summary[content_perspective]).keys()]

            # 主题词去重
            content_perspective_list = list(set(content_perspective_list))

            example_input = """
                [
                    'CEEMDAN (Complete Ensemble Empirical Mode Decomposition with Adaptive Noise)',
                    'Improved TFR (Time-Frequency Representation) demodulation',
                    'KC (Kurtosis-Correlation) indicator',
                    'CWT (Continuous Wavelet Transform)',
                    'Simulation and experimental validation (IMS/CWRU datasets)',
                    'Resonance demodulation comparison',
                    'EMD (Empirical Mode Decomposition) analysis',
                    'Experimental approach for FOWT analysis',
                    'Free-decay tests for resonance properties',
                    'Regular wave tests for RAOs',
                    'Wind-wave combined tests for dynamic performance',
                    'Wavelet transform for time-frequency analysis',
                    'Mooring system simulation for tension force analysis',
                    'Froude scaling for model testing',
                    'Optical tracking system for motion capture',
                    'Extended Kalman Filter (EKF) bank structured according to GOS',
                    'Nonlinear model for RSC and linear model for GSC',
                    'Clarke transformation for reference frame conversion',
                    'Page-Hinkley (P-H) statistical test for fault detection',
                    "Jacobian's discrete-time method for nonlinearity",
                    'Logical combination for fault localization',
                    'Bank of Luenberger observers for fault diagnosis',
                    'Sliding mode observers for fault detection',
                    'Sensitivity analysis for failure rates',
                    'Cost-based component comparison parameter',
                    'Monte Carlo simulation for lifecycle maintenance',
                    'Statistical waiting time tool for downtime prediction',
                    'Failure rate classification (global/local)',
                    'Fatigue strength scaling theory application',
                    'Data collection and analysis from SCADA systems',
                    'Machine learning algorithms for fault diagnosis',
                    'Life-cycle cost analysis for wind energy projects',
                    'Decision support systems for O&M decision-making',
                    'Signal processing techniques for fault detection',
                    'Deep learning models for wind speed forecasting',
                    'Quantitative risk assessment in wind energy',
                    'Sensitivity analysis for wind resource assessment',
                    'IoT-based platforms for wind turbine monitoring',
                    'Artificial neural networks for O&M optimization',
                    'Digital twin technology for wind turbine simulation'
                ]
            """

            example_output = """
                {
                  'Signal Decomposition and Time-Frequency Analysis Methods': {
                    'CEEMDAN (Complete Ensemble Empirical Mode Decomposition with Adaptive Noise)': 10,
                    'Improved TFR (Time-Frequency Representation) demodulation': 10,
                    'CWT (Continuous Wavelet Transform)': 10,
                    'EMD (Empirical Mode Decomposition) analysis': 10,
                    'Wavelet transform for time-frequency analysis': 10,
                    'Resonance demodulation comparison': 9,
                    'Signal processing techniques for fault detection': 8
                  },
                  'Statistical Analysis and Fault Detection Metrics': {
                    'Page-Hinkley (P-H) statistical test for fault detection': 10,
                    'Monte Carlo simulation for lifecycle maintenance': 10,
                    'Statistical waiting time tool for downtime prediction': 10,
                    'KC (Kurtosis-Correlation) indicator': 9,
                    'Sensitivity analysis for failure rates': 9,
                    'Failure rate classification (global/local)': 9,
                    'Quantitative risk assessment in wind energy': 9,
                    'Sensitivity analysis for wind resource assessment': 9
                  },
                  'Experimental and Simulation-Based Validation Techniques': {
                    'Simulation and experimental validation (IMS/CWRU datasets)': 10,
                    'Experimental approach for FOWT analysis': 10,
                    'Free-decay tests for resonance properties': 10,
                    'Regular wave tests for RAOs': 10,
                    'Wind-wave combined tests for dynamic performance': 10,
                    'Mooring system simulation for tension force analysis': 10,
                    'Froude scaling for model testing': 10,
                    'Optical tracking system for motion capture': 10
                  },
                  'Modeling and Observer-Based Diagnostic Methods': {
                    'Extended Kalman Filter (EKF) bank structured according to GOS': 10,
                    'Nonlinear model for RSC and linear model for GSC': 10,
                    'Bank of Luenberger observers for fault diagnosis': 10,
                    'Sliding mode observers for fault detection': 10,
                    'Clarke transformation for reference frame conversion': 9,
                    'Jacobian's discrete-time method for nonlinearity': 9,
                    'Logical combination for fault localization': 8,
                  },
                  'Data-Driven and Machine Learning Approaches': {
                    'Machine learning algorithms for fault diagnosis': 10,
                    'Deep learning models for wind speed forecasting': 10,
                    'Artificial neural networks for O&M optimization': 10,
                    'Data collection and analysis from SCADA systems': 9,
                    'IoT-based platforms for wind turbine monitoring': 9,
                    'Digital twin technology for wind turbine simulation': 9
                  },
                  'Cost and Lifecycle Analysis Methods': {
                    'Life-cycle cost analysis for wind energy projects': 10,
                    'Cost-based component comparison parameter': 10,
                    'Decision support systems for O&M decision-making': 9,
                    'Fatigue strength scaling theory application': 8
                  }
                }
            """

            system_prompt = f"Please categorize the keywords on {content_perspective.replace('_', ' ')} of academic papers provided by the user and generate high-representative keyword for each category, and rate the relevance of each keyword to the high-representative keyword on a ten-point scale, and please check and ensure that no keyword classification has been missed (all of them in JSON format) {{Example Input: {example_input}}} {{Example Output: {example_output}}}"
            user_prompt = str(content_perspective_list)

            max_retires = 5
            theme_dict = {}
            for _ in range(max_retires):
                theme_dict = APILLMParser.call_llm_api(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    resp_format="json"
                )

                if isinstance(theme_dict, dict) and len(theme_dict) > 0:
                    break

            total_theme_dict[content_perspective] = theme_dict

            ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'],
                                               new_message=f"{content_perspective} 已生成")

        characterize_id = RandomStrGenerator.generate_uuid()

        mysql_result3 = ScienceResearchMapper.insert_theme_characterize({
            "characterize_id": characterize_id,
            "user_id": self.knowledge_finding_config['user_id'],
            "task_id": self.knowledge_finding_config['task_id'],
            "research_questions": str(total_theme_dict['research_questions']),
            "research_methods": str(total_theme_dict['research_methods']),
            "research_contributions": str(total_theme_dict['research_contributions']),
            "research_limitations": str(total_theme_dict['research_limitations'])
        })

        ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'], new_message="生成完毕")

    def summarize_paper_themes(self):

        ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'], new_message="开始总结论文")

        mysql_result = ScienceResearchMapper.select_paper_search_where_user_id_and_task_id({
            "user_id": self.knowledge_finding_config['user_id'],
            "task_id": self.knowledge_finding_config['task_id']
        })

        paper_list = mysql_result.get_data_on_results()

        # 解析pdf获取论文文本
        try:
            for m, paper in enumerate(paper_list):

                paper_title = paper['paper_title']

                try:

                    if self.knowledge_finding_config['task_type'] == 'retrieval':
                        hashed_paper_title = HashParser.hash_encode(paper_title)
                        pdf_file_path = f"{self.knowledge_finding_config['paper_pdf_path']}/{hashed_paper_title}.pdf"

                    else:
                        mysql_result0 = KnowledgeBaseMapper.select_file_where_file_name({
                            "file_name": paper_title
                        })

                        if not mysql_result0.verify_data_on_results():
                            continue

                        file_id = mysql_result0.get_data_on_results()[0]['file_id']
                        pdf_file_path = f"{self.knowledge_finding_config['paper_pdf_path']}/{file_id}.pdf"

                    # 判断该论文是否存在pdf文件
                    if not os.path.exists(pdf_file_path):
                        # 跳过读取
                        continue

                    # 判断该论文是否已经生成全部维度的主题词
                    if paper['research_questions'] and paper['research_questions_explaining'] and paper['research_methods'] and paper['research_methods_explaining'] and paper['research_contributions'] and paper['research_contributions_explaining'] and paper['research_limitations'] and paper['research_limitations_explaining'] and paper['paper_summary'] and paper['paper_zh_summary']:
                        # 跳过读取
                        continue

                    try:
                        paper_text = PDFParser.load(pdf_file_path)
                    except Exception as e:
                        # 下载失败，说明PDF文件有问题，将删除
                        ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'],
                                                           new_message=f"下载PDF文件失败: {paper_title}")
                        try:
                            os.remove(pdf_file_path)
                        except Exception as e:
                            print(e)

                        continue

                    ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'], new_message=f"已读取pdf文件: {paper_title}")

                    print(f"文件约为: {int(APILLMParser.calculate_token(paper_text)/1000) + 1}k Tokens")
                    print("""
                        Deepseek 上下文长度: 64k Tokens
                        GLM-4-Long 上下文长度: 1000k Tokens
                    """)

                    # 分割文本为段落

                    paragraph_list = []
                    max_str = 8000
                    current_paragraph = ""
                    for sentence in paper_text.split("\n"):
                        if len(current_paragraph) + len(sentence) > max_str:
                            paragraph_list.append(current_paragraph)
                            current_paragraph = ""

                        current_paragraph += sentence + "\n"
                    else:
                        paragraph_list.append(current_paragraph)

                    del paper_text

                    ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'], new_message=f"已分割文本: {paper_title}")

                    example_output = """
                        {
                          "research_questions_keywords": {
                            "Abnormal data identification in SCADA systems": "The paper focuses on detecting anomalies in wind turbine SCADA data, which are critical for early fault warnings but challenging due to noise, imbalances, and high dimensionality.",
                            "Impact of modeling data quality on fault prediction": "The study argues that poor-quality training data (e.g., containing anomalies) degrades fault prediction models, necessitating robust preprocessing.",
                            "Handling high-dimensional and imbalanced operational data": "SCADA data’s complexity and wind speed imbalance (more low-speed data) require specialized methods like I-SAE and density ratio-based clustering.",
                            "Separability enhancement between normal and abnormal data": "The proposed I-SAE dynamically filters training samples to amplify differences in reconstruction errors, improving anomaly detection.",
                            "Integration of unsupervised learning with prior knowledge": "Combines unsupervised techniques (e.g., autoencoders) with domain-specific relationships (e.g., wind speed-power curves) to refine anomaly identification."
                          },
                          "research_methods_keywords": {
                            "Improved Stacked Autoencoder (I-SAE) with partial data reconstruction": "Enhances anomaly detection by selectively training on low-error data and using partial reconstruction to sharpen separability.",
                            "Density ratio-based DBSCAN clustering": "Addresses wind speed imbalance by normalizing cluster density relative to wind speed intervals, reducing misclassification.",
                            "Otsu algorithm for stacked anomaly detection": "Segments stacked anomalies (e.g., fault-induced clusters) by maximizing inter-class variance in low-dimensional variable relationships.",
                            "Serialization ensemble algorithm (voting criteria)": "Merges results from multiple I-SAE models with varying thresholds via voting, improving robustness.",
                            "Exponentially Weighted Moving Average (EWMA) smoothing": "Reduces noise in reconstruction errors to stabilize fault alerts.",
                          },
                          "research_contributions_keywords": {
                            "Unsupervised sequential-ensemble framework for data processing": "Eliminates reliance on labeled data by integrating autoencoders, clustering, and domain knowledge.",
                            "Integration of high-dimensional features and low-dimensional relationships": "Combines deep learning (SAE) with physics-based variable relationships for comprehensive anomaly processing.",
                            "Density ratio adaptation for imbalanced wind speed data": "Resolves wind speed data imbalance issues, improving sparse anomaly detection.",
                            "Improved fault early warning performance through data quality": "Demonstrates that high-quality modeling data reduces false alarms and enables earlier fault detection (e.g., 4-day advance warning in case studies).",
                            "Robustness via voting criteria and parameter ensembles": "Mitigates parameter sensitivity by aggregating outputs from multiple models.",
                          },
                          "research_limitations_keywords": {
                            "Dependency on specific wind turbine datasets (limited generalization)": "Evaluated on a specific wind farm’s SCADA data; generalization to other turbine types or environments is unverified.",
                            "Computational complexity of ensemble models": "Ensemble methods (e.g., multiple I-SAE models) may hinder real-time deployment.",
                            "Assumption of normal data dominance for reconstruction errors": "Relies on normal data being the majority for reconstruction-based methods, which may fail in skewed scenarios.",
                            "Limited evaluation of real-time processing feasibility": "Focuses on offline modeling; online implementation challenges (e.g., latency) are not addressed.",
                            "Manual tuning of parameters (e.g., density ratio thresholds)": "Requires expert intervention for thresholds (e.g., density ratio, Otsu partitions)."
                          }
                        }
                    """

                    base_messages = []
                    summary_messages = []

                    if len(paragraph_list) == 1:
                        prompt = f"summarize this whole paper provided by the user into at least 6 research questions keywords, research methods keywords, research contributions keywords and research limitations keywords, and provide a short paragraph of reasoning of each keyword to the paper in English. (all of them in JSON format) {{Example Output: {example_output}}}"
                        summary_messages = [
                            {
                                "role": "user",
                                "content": f"Please {prompt}: [{paragraph_list[0]}]"
                            }
                        ]

                    else:

                        for i, paragraph in enumerate(paragraph_list):

                            base_messages.append({
                                "role": "user",
                                "content": f"This is the part {i} of the multiple input. There will be more content to follow. Please wait for the complete input before generating the final response: [{paragraph}]"
                            })
                            base_messages.append({
                                "role": "assistant",
                                "content": "ok."
                            })
                        else:
                            prompt = f"summarize this whole paper provided by the user into at least 6 research questions keywords, research methods keywords, research contributions keywords and research limitations keywords, and provide a short paragraph of reasoning of each keyword to the paper in English. (all of them in JSON format) {{Example Output: {example_output}}}"
                            summary_messages = base_messages + [
                                {
                                    "role": "user",
                                    "content": f"All the chunks have been inputted. Please combine all the previous chunks to {prompt}"
                                }
                            ]

                    max_retries = 5
                    new_paper_summary_dict = {}
                    for _ in range(max_retries):

                        paper_summary_dict = APILLMParser.call_llm_api(
                            messages=summary_messages,
                            resp_format="json",
                            model_type="zhipu",
                            model="glm-4-long",
                            will_clear_invalid_char=True
                        )

                        # 修正总结的键名
                        new_paper_summary_dict = {}
                        for k, v in paper_summary_dict.items():

                            if 'research_question' in k.lower():
                                new_paper_summary_dict['research_questions'] = v
                            elif 'research_method' in k.lower():
                                new_paper_summary_dict['research_methods'] = v
                            elif 'research_contribution' in k.lower():
                                new_paper_summary_dict['research_contributions'] = v
                            elif 'research_limitation' in k.lower():
                                new_paper_summary_dict['research_limitations'] = v

                        if len(new_paper_summary_dict) >= 4 and len(new_paper_summary_dict['research_questions']) >= 4 and len(new_paper_summary_dict['research_methods']) >= 4 and len(new_paper_summary_dict['research_contributions']) >= 4 and len(new_paper_summary_dict['research_limitations']) >= 4:
                            break

                    ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'], new_message=f"已读取文本总结: {paper_title}")

                    if len(paragraph_list) == 1:
                        prompt = f"summarize this whole paper based on the following keywords: {new_paper_summary_dict}"
                        model_messages = [
                            {
                                "role": "user",
                                "content": f"Please {prompt}"
                            }
                        ]
                    else:
                        prompt = f"summarize the above paper based on the following keywords: {new_paper_summary_dict}"
                        model_messages = base_messages + [
                            {
                                "role": "user",
                                "content": f"All the chunks have been inputted. Please combine all the previous chunks to {prompt}"
                            }
                        ]

                    paper_summary_text = APILLMParser.call_llm_api(
                        messages=model_messages,
                        model_type="zhipu",
                        model="glm-4-long",
                        will_clear_invalid_char=True
                    )

                    new_paper_summary_dict["paper_summary"] = paper_summary_text

                    # 翻译论文总结
                    paper_zh_summary = TextTranslator.translate(new_paper_summary_dict["paper_summary"], translate_source="deepseek").get_data_on_results()

                    # 使用正则表达式去除括号及括号内的内容
                    paper_zh_summary = re.sub(r'（注：[^）]*）', '', paper_zh_summary)

                    # 计算主题词与论文之间的权重(余弦相似度)

                    # 获取每个新的主题词的高维特征向量

                    paper_summary_text_embedding = SemanticVectorCalculation.calculate_text_group([paper_summary_text])[0]

                    research_questions_keyword_list = [x for x in new_paper_summary_dict['research_questions'].keys()]
                    research_questions_embedding_list = SemanticVectorCalculation.calculate_text_group(research_questions_keyword_list)
                    research_questions_theme_dict = {}
                    for i, embedding in enumerate(research_questions_embedding_list):
                        max_similarity = DistanceCalculation.calculate_cosine_similarity(embedding, paper_summary_text_embedding)
                        # 权重分数保留两位小数
                        score = round((max_similarity + 1) / 2.0, 2)
                        research_questions_theme_dict[research_questions_keyword_list[i]] = score

                    research_methods_keyword_list = [x for x in new_paper_summary_dict['research_methods'].keys()]
                    research_methods_embedding_list = SemanticVectorCalculation.calculate_text_group(research_methods_keyword_list)
                    research_methods_theme_dict = {}
                    for i, embedding in enumerate(research_methods_embedding_list):
                        max_similarity = DistanceCalculation.calculate_cosine_similarity(embedding, paper_summary_text_embedding)
                        # 权重分数保留两位小数
                        score = round((max_similarity + 1) / 2.0, 2)
                        research_methods_theme_dict[research_methods_keyword_list[i]] = score

                    research_contributions_keyword_list = [x for x in new_paper_summary_dict['research_contributions'].keys()]
                    research_contributions_embedding_list = SemanticVectorCalculation.calculate_text_group(research_contributions_keyword_list)
                    research_contributions_theme_dict = {}
                    for i, embedding in enumerate(research_contributions_embedding_list):
                        max_similarity = DistanceCalculation.calculate_cosine_similarity(embedding, paper_summary_text_embedding)
                        # 权重分数保留两位小数
                        score = round((max_similarity + 1) / 2.0, 2)
                        research_contributions_theme_dict[research_contributions_keyword_list[i]] = score

                    research_limitations_keyword_list = [x for x in new_paper_summary_dict['research_limitations'].keys()]
                    research_limitations_embedding_list = SemanticVectorCalculation.calculate_text_group(research_limitations_keyword_list)
                    research_limitations_theme_dict = {}
                    for i, embedding in enumerate(research_limitations_embedding_list):
                        max_similarity = DistanceCalculation.calculate_cosine_similarity(embedding, paper_summary_text_embedding)
                        # 权重分数保留两位小数
                        score = round((max_similarity + 1) / 2.0, 2)
                        research_limitations_theme_dict[research_limitations_keyword_list[i]] = score

                    # example_output = """
                    #     graph LR
                    #     A[Paper: Vibration Analysis of Wind Turbine Transmission Systems with Gear Eccentricity, Tooth Root Crack, and Bearing Faults] --> B[Objectives]
                    #     A --> C[Methodology]
                    #     A --> D[Key Contributions]
                    #     A --> E[Results]
                    #     A --> F[Limitations]
                    #
                    #     B --> B1[Investigate vibration characteristics under faults]
                    #     B --> B2[Establish theoretical framework for vibration analysis]
                    #     B --> B3[Develop dynamic model for fault simulation]
                    #     B --> B4[Identify diagnostic criteria for faults]
                    #     B --> B5[Analyze fault interaction and propagation]
                    #
                    #     C --> C1[Enhanced Mesh Stiffness Calculation]
                    #     C1 --> C1a[Helical gears with root cracks]
                    #     C1 --> C1b[Time-varying stiffness computation]
                    #
                    #     C --> C2[Dynamic Modeling]
                    #     C2 --> C2a[22-DOF lumped parameter model]
                    #     C2 --> C2b[Helical gear-rotor-bearing system]
                    #
                    #     C --> C3[Simulation]
                    #     C3 --> C3a[Runge-Kutta method for solving equations]
                    #     C3 --> C3b[Vibration response simulation]
                    #
                    #     C --> C4[Fault Evaluation]
                    #     C4 --> C4a[Statistical indicators: RMS, Kurtosis]
                    #     C4 --> C4b[Severity assessment of gear eccentricity/cracks]
                    #
                    #     C --> C5[Fault Types]
                    #     C5 --> C5a[Gear eccentricity]
                    #     C5 --> C5b[Tooth root crack]
                    #     C5 --> C5c[Bearing faults]
                    #
                    #     D --> D1[Theoretical framework for fault vibration analysis]
                    #     D --> D2[Dynamic model for fault scenario simulation]
                    #     D --> D3[Diagnostic criteria for gearbox faults]
                    #     D --> D4[Insights into fault interaction mechanisms]
                    #
                    #     E --> E1[Identified vibration features as diagnostic markers]
                    #     E --> E2[Model validated for fault response simulation]
                    #     E --> E3[Statistical indicators correlate with fault severity]
                    #
                    #     F --> F1[Complexity in coupled fault analysis]
                    #     F --> F2[Assumptions in fault modeling]
                    #     F2 --> F2a[Example: idealized conditions]
                    #     F --> F3[Limited experimental validation]
                    #     F --> F4[Need for improved diagnostic accuracy]
                    #
                    #     style A fill:#f9f,stroke:#333,stroke-width:4px
                    #     classDef methodology fill:#9cf,stroke:#333;
                    #     class C,C1,C2,C3,C4,C5 methodology;
                    #     classDef results fill:#cfc,stroke:#333;
                    #     class E,E1,E2,E3 results;
                    #     classDef limitations fill:#fcc,stroke:#333;
                    #     class F,F1,F2,F3,F4 limitations;
                    # """
                    #
                    # system_prompt = f"Please provide a comprehensive, detailed and professional description of the paper offered by user in Mermaid syntax. {{Example Output: {example_output}}}"
                    # user_prompt = paper_summary_text
                    #
                    # for _ in range(max_retries):
                    #     try:
                    #         paper_model_structure_mermaid = APILLMParser.call_llm_api(
                    #             system_prompt=system_prompt,
                    #             user_prompt=user_prompt,
                    #             model_type="zhipu",
                    #             model="glm-4-plus",
                    #             will_clear_invalid_char=True
                    #         )
                    #
                    #         # 解析mermaid格式文本
                    #         first_match = re.search(r'```mermaid(.*?)```', paper_model_structure_mermaid, re.DOTALL)
                    #         if first_match:
                    #             paper_model_structure_mermaid = first_match.group(1).strip()
                    #         else:
                    #             continue
                    #
                    #         mermaid_prompt = f"根据以下Mermaid语言绘制流程图，图片的清晰度要高，如300dpi: {paper_model_structure_mermaid}"
                    #         completion_list = APILLMParser.call_assistant_llm_api(text=mermaid_prompt)
                    #
                    #         image_dict = None
                    #         for assistant in completion_list:
                    #             print(assistant)
                    #             if hasattr(assistant, 'choices') and len(assistant.choices) > 0:
                    #                 choice = assistant.choices[0]
                    #                 if hasattr(choice, 'delta'):
                    #                     delta = choice.delta
                    #                     if hasattr(delta, 'tool_calls') and len(delta.tool_calls) > 0:
                    #                         tool_call = delta.tool_calls[0]
                    #                         if hasattr(tool_call, 'function'):
                    #                             function = tool_call.function
                    #                             if hasattr(function,
                    #                                        'outputs') and function.outputs is not None and len(
                    #                                     function.outputs) > 0:
                    #                                 function_output = function.outputs[0]
                    #                                 if hasattr(function_output, 'content'):
                    #                                     image_dict = function_output.content
                    #                                     break
                    #
                    #         if image_dict is not None:
                    #             image_dict = eval(image_dict)
                    #             if 'url' in image_dict.keys():
                    #                 mermaid_image_url = image_dict['url']
                    #
                    #                 mermaid_save_path = f"{self.knowledge_finding_config['mermaid_image_path']}/paper_mermaid.png"
                    #
                    #                 # 下载mermaid流程图
                    #                 # 发送 HTTP GET 请求
                    #                 response = requests.get(mermaid_image_url, stream=True)
                    #                 response.raise_for_status()  # 检查请求是否成功
                    #
                    #                 # 以二进制写入模式保存图片
                    #                 with open(mermaid_save_path, 'wb') as file:
                    #                     for chunk in response.iter_content(1024):
                    #                         file.write(chunk)
                    #
                    #                 ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'],
                    #                                                    new_message="Mermaid流程图成功保存")
                    #
                    #                 break
                    #             else:
                    #                 continue
                    #
                    #     except Exception as e:
                    #         print(e)
                    #         continue

                    # 保存论文主题词到数据库

                    mysql_result1 = ScienceResearchMapper.update_paper({
                        "research_questions": str(research_questions_theme_dict),
                        "research_questions_explaining": str(new_paper_summary_dict['research_questions']),
                        "research_methods": str(research_methods_theme_dict),
                        "research_methods_explaining": str(new_paper_summary_dict['research_methods']),
                        "research_contributions": str(research_contributions_theme_dict),
                        "research_contributions_explaining": str(new_paper_summary_dict['research_contributions']),
                        "research_limitations": str(research_limitations_theme_dict),
                        "research_limitations_explaining": str(new_paper_summary_dict['research_limitations']),
                        "paper_summary": new_paper_summary_dict["paper_summary"],
                        "paper_zh_summary": paper_zh_summary,
                        "paper_id": paper['paper_id']
                    })

                    ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'], new_message=f"{m}.已总结论文: {paper_title}")

                    del paragraph_list
                    del paper
                    del paper_summary_dict

                except Exception as e:
                    ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'], new_message=f"{m}.{paper_title}总结失败: {e}")

        except Exception as e:

            ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'], new_message=f"{e}")

        ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'], new_message="生成完成")

    def search_papers_on_themes_zh(self):

        # TODO(已弃用)

        APILLMParser.search_paper_call_zhipu_llm_api(
            keywords=self.knowledge_finding_config['search_words']
        )

        request_get_params = {
            "query": self.knowledge_finding_config['search_words'],
            "fields": "title,authors,publicationDate,abstract,url,externalIds,journal,openAccessPdf",
            "publicationTypes": "JournalArticle",
            "openAccessPdf": "",  # 这是一个布尔标志，值为空表示 True
            "limit": 10
        }

        resp = RequestSender.send_request_with_retry(
            url="https://api.semanticscholar.org/graph/v1/paper/search",
            params=request_get_params
        )
        paper_result = resp['data']

        # 对所有论文的字段格式进行修正
        for paper in paper_result:
            paper['externalIds'] = paper['externalIds']['DOI']
            paper['openAccessPdf'] = paper['openAccessPdf']['url']
            paper['journal'] = paper['journal']['name'] if 'name' in paper['journal'].keys() else ""
            paper['authors'] = ",".join([author['name'] if 'name' in author.keys() else "" for author in paper['authors']])

        print("检索完成")

        # 将检索结果保存到数据库
        for paper in paper_result:
            # 生成新的id
            paper_id = RandomStrGenerator.generate_uuid()

            mysql_result = ScienceResearchMapper.select_paper_where_title({
                "paper_title": paper['title']
            })

            if not mysql_result.verify_data_on_results():

                try:

                    mysql_result1 = ScienceResearchMapper.insert_paper({
                        "paper_id": paper_id,
                        "paper_owner": self.knowledge_finding_config['user_id'],
                        "paper_belong_task": self.knowledge_finding_config['task_id'],
                        "paper_search_words": self.knowledge_finding_config['search_words'],
                        "paper_title": paper['title'],
                        "paper_authors": paper['authors'],
                        "paper_publish_time": paper['publicationDate'],
                        "paper_abstract": paper['abstract'],
                        "paper_url": paper['url'],
                        "paper_doi": paper['externalIds'],
                        "paper_journal": paper['journal'],
                        "paper_pdf_url": paper['openAccessPdf'],
                        "paper_language": "en"
                    })

                except Exception as e:
                    print(f"{paper['title']} 保存出错")

            else:

                print(f"{paper['title']} 已存在")

        print("保存完成")

    def search_papers_on_themes(self):

        ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'], new_message="开始检索")

        if self.knowledge_finding_config['task_type'] == 'retrieval':

            # 单次查询最多返回100条，分割查询
            group_num = 100
            total_paper_num = self.knowledge_finding_config["paper_num"]
            send_turn = total_paper_num // group_num
            last_paper_num = total_paper_num % group_num
            paper_num_list = [group_num] * send_turn + [last_paper_num] if last_paper_num != 0 else [group_num] * send_turn

            paper_result = []
            paper_title_list = []
            for i in range(len(paper_num_list)):
                request_get_params = {
                    "query": self.knowledge_finding_config['search_words'],
                    "fields": "title,authors,publicationDate,abstract,url,externalIds,journal,openAccessPdf",
                    "publicationTypes": "JournalArticle",  # 限期刊论文
                    "openAccessPdf": "",  # 有pdf链接的论文
                    "limit": paper_num_list[i],
                    "offset": 0 if i == 0 else sum(paper_num_list[:i])
                }

                time.sleep(5)

                result = RequestSender.send_request_with_retry(
                    url="https://api.semanticscholar.org/graph/v1/paper/search",
                    params=request_get_params
                )

                if not result['check']:
                    print(result['message'])
                    continue

                resp = result['data']

                ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'], new_message=f"可检索论文数量: {resp['total']}")

                # 去重论文
                for paper in resp['data']:
                    if paper['title'] is None or paper['title'] in ["", "\n", " "] or paper['title'] in paper_title_list:
                        continue

                    paper_result.append(paper)
                    paper_title_list.append(paper['title'])

                if i < len(paper_num_list) - 1:
                    time.sleep(10)

                # 查询数量不能超过API能够检索的最大数量
                if resp['total'] < self.knowledge_finding_config["paper_num"] and resp['total'] < sum(paper_num_list[:(i + 1)]):
                    self.knowledge_finding_config["paper_num"] = resp['total']
                    break

            # 对所有论文的字段格式进行修正
            for paper in paper_result:
                paper['externalIds'] = paper['externalIds']['DOI']
                paper['openAccessPdf'] = paper['openAccessPdf']['url']
                paper['journal'] = paper['journal']['name'] if 'name' in paper['journal'].keys() else ""
                paper['authors'] = ",".join([author['name'] if 'name' in author.keys() else "" for author in paper['authors']])

            ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'], new_message="检索完成")

        else:

            mysql_result0 = KnowledgeBaseMapper.select_file_where_belong_folder({
                "belong_folder": self.knowledge_finding_config['paper_pdf_path'].split("/")[-1]
            })

            paper_result = [{
                "title": record['file_name'],
                "abstract": "",
                "authors": "",
                "publicationDate": "",
                "url": "",
                "externalIds": "",
                "journal": "",
                "openAccessPdf": ""
            } for record in mysql_result0.get_data_on_results()]

        # 将检索结果保存到数据库
        for i, paper in enumerate(paper_result):

            mysql_result = ScienceResearchMapper.select_paper_where_title({
                "paper_title": paper['title']
            })

            # 判断论文是否已经存在于数据库中
            if not mysql_result.verify_data_on_results():

                try:

                    # 生成论文id
                    paper_id = RandomStrGenerator.generate_uuid()

                    # 翻译标题
                    if paper['title'] is None or paper['title'] == "":
                        paper_zh_title = ""
                    else:
                        paper_zh_title = TextTranslator.translate(paper['title'], translate_source="deepseek").get_data_on_results()

                    # 使用正则表达式去除括号及括号内的内容
                    paper_zh_title = re.sub(r'（注：[^）]*）', '', paper_zh_title)

                    # 翻译摘要
                    if paper['abstract'] is None or paper['abstract'] == "":
                        paper_zh_abstract = ""
                    else:
                        paper_zh_abstract = TextTranslator.translate(paper['abstract'], translate_source="deepseek").get_data_on_results()

                    # 使用正则表达式去除括号及括号内的内容
                    paper_zh_abstract = re.sub(r'（注：.*?）', '', paper_zh_abstract)

                    # 保存论文数据到数据库
                    mysql_result1 = ScienceResearchMapper.insert_paper({
                        "paper_id": paper_id,
                        "paper_title": paper['title'],
                        "paper_zh_title": paper_zh_title,
                        "paper_authors": paper['authors'],
                        "paper_publish_time": paper['publicationDate'],
                        "paper_abstract": paper['abstract'],
                        "paper_zh_abstract": paper_zh_abstract,
                        "paper_url": paper['url'],
                        "paper_doi": paper['externalIds'],
                        "paper_journal": paper['journal'],
                        "paper_pdf_url": paper['openAccessPdf'],
                        "paper_language": "en"
                    })

                    # 保存查询数据到数据库
                    mysql_result2 = ScienceResearchMapper.update_paper_search_set_paper_id({
                        "paper_id": paper_id,
                        "search_id": self.knowledge_finding_config['search_id_list'][i]
                    })

                except Exception as e:
                    ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'],
                                                       new_message=f"{i}: {paper['title']} 保存出错: {e}")
            else:

                paper_id = mysql_result.get_data_on_results()[0]['paper_id']

                # 保存查询数据到数据库
                mysql_result3 = ScienceResearchMapper.update_paper_search_set_paper_id({
                    "paper_id": paper_id,
                    "search_id": self.knowledge_finding_config['search_id_list'][i]
                })

                ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'],
                                                   new_message=f"{i}: {paper['title']} 已存在")

            ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'], new_message=f"{i}: {paper['title']} 保存完成")

            del paper

        ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'], new_message="全部保存完成")

    def download_paper_pdf(self):

        ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'], new_message="开始下载pdf")

        if self.knowledge_finding_config['task_type'] != 'retrieval':
            ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'], new_message="已存在pdf文件无需下载")
            return

        # TODO: 无VPN，一般网络下载PDF非常慢

        # 查询数据库获取所有论文
        mysql_result = ScienceResearchMapper.select_paper_search_where_user_id_and_task_id({
            "user_id": self.knowledge_finding_config['user_id'],
            "task_id": self.knowledge_finding_config['task_id']
        })
        paper_result_list = mysql_result.get_data_on_results()

        # 下载论文pdf
        for paper in paper_result_list:
            # 对论文名称进行哈希加密
            try:
                hashed_paper_title = HashParser.hash_encode(paper['paper_title'])
                pdf_file_path = f"{self.knowledge_finding_config['paper_pdf_path']}/{hashed_paper_title}.pdf"

                # 若该论文pdf已下载，则跳过
                if os.path.exists(pdf_file_path):
                    continue
            except Exception as e:
                print(e)
                continue

            status, message = RequestSender.download_pdf_file(
                pdf_url=paper['paper_pdf_url'],
                file_path=pdf_file_path
            )

            ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'], new_message=message)

            del message
            del paper

        # 删除无效的pdf文件
        for file_name in os.listdir(self.knowledge_finding_config['paper_pdf_path']):
            file_path = f"{self.knowledge_finding_config['paper_pdf_path']}/{file_name}"
            if not file_name.endswith(".pdf"):
                # 删除非pdf后缀的文件
                os.remove(file_path)
            elif os.path.getsize(file_path) / 1024 < 20:
                # 删除文件内存小于20KB的文件
                os.remove(file_path)

        ScienceResearchProgress.update_log(user_id=self.knowledge_finding_config['user_id'], new_message="下载完成")

    def summarize_paper_pdf(self):

        print("开始总结论文")

        # 解析pdf获取论文文本
        try:
            for m, file_name in enumerate(os.listdir(self.knowledge_finding_config['paper_pdf_path'])):
                try:

                    pdf_file_path = f"{self.knowledge_finding_config['paper_pdf_path']}/{file_name}"
                    paper_text = PDFParser.load(pdf_file_path)

                    print(f"已读取pdf文件: {file_name}")

                    # 分割文本为段落

                    paragraph_list = []
                    max_str = 4000
                    current_paragraph = ""
                    for sentence in paper_text.split("\n"):
                        if len(current_paragraph) + len(sentence) > max_str:
                            paragraph_list.append(current_paragraph)
                            current_paragraph = ""

                        current_paragraph += sentence + "\n"

                    print(f"已分割文本: {file_name}")

                    base_messages = []
                    summary_messages = []
                    for i, paragraph in enumerate(paragraph_list):
                        if i == len(paragraph_list) - 1:
                            prompt = "summarize this whole paper provided by the user into research questions, research methods, research contributions and research limitations in JSON format."
                            summary_messages = base_messages + [
                                {
                                    "role": "user",
                                    "content": f"This is the last part of the multiple input. Please combine all the previous chunks to {prompt}: [{paragraph}]"
                                }
                            ]
                        else:
                            base_messages.append({
                                "role": "user",
                                "content": f"This is the part {i} of the multiple input. There will be more content to follow. Please wait for the complete input before generating the final response: [{paragraph}]"
                            })
                            base_messages.append({
                                "role": "assistant",
                                "content": "ok."
                            })

                    max_retries = 5
                    new_paper_summary_dict = {}
                    for _ in range(max_retries):

                        paper_summary_dict = APILLMParser.call_llm_api(
                            messages=summary_messages,
                            resp_format="json",
                            model_type="zhipu",
                            model="glm-4-long",
                            will_clear_invalid_char=True
                        )

                        # 修正总结的键名
                        new_paper_summary_dict = {}
                        for k, v in paper_summary_dict.items():
                            if k.lower() == 'research_questions' or f"{k}s" == 'research_questions':
                                new_paper_summary_dict['research_questions'] = str(v)
                            elif k.lower() == 'research_methods' or f"{k}s" == 'research_methods':
                                new_paper_summary_dict['research_methods'] = str(v)
                            elif k.lower() == 'research_contributions' or f"{k}s" == 'research_contributions':
                                new_paper_summary_dict['research_contributions'] = str(v)
                            elif k.lower() == 'research_limitations' or f"{k}s" == 'research_limitations':
                                new_paper_summary_dict['research_limitations'] = str(v)

                        if len(new_paper_summary_dict) >= 4:
                            break

                    print(f"已读取文本总结: {file_name}")

                    prompt = "provide a comprehensive, detailed and professional description on how the model operates proposed in this paper."
                    model_messages = base_messages + [
                        {
                            "role": "user",
                            "content": f"This is the last part of the multiple input. Please combine all the previous chunks to {prompt}."
                        }
                    ]

                    paper_model_structure = APILLMParser.call_llm_api(
                        messages=model_messages,
                        model_type="zhipu",
                        model="glm-4-long",
                        will_clear_invalid_char=True
                    )
                    new_paper_summary_dict["paper_summary"] = paper_model_structure

                    # 获取论文标题对应的论文id
                    mysql_result = ScienceResearchMapper.select_paper_where_id({
                        "paper_id": file_name.rstrip(".pdf")
                    })

                    paper_id_result = mysql_result.get_data_on_results()
                    
                    if len(paper_id_result) == 0:
                        print(f"{m+1}.{file_name}总结失败")
                        continue

                    paper_id = paper_id_result[0].paper_id

                    # 保存到数据库
                    mysql_result1 = ScienceResearchMapper.insert_paper_summaries({
                        "paper_id": paper_id,
                        "research_questions": new_paper_summary_dict['research_questions'].replace("'", "''"),
                        "research_methods": new_paper_summary_dict['research_methods'].replace("'", "''"),
                        "research_contributions": new_paper_summary_dict['research_contributions'].replace("'", "''"),
                        "research_limitations": new_paper_summary_dict['research_limitations'].replace("'", "''"),
                        "paper_summary": new_paper_summary_dict["paper_summary"].replace("'", "''")
                    })

                    print(f"{m+1}.已总结论文: {file_name}")

                except Exception as e:
                    print(e)
                    print(f"{m+1}.{file_name}总结失败")

        except Exception as e:

            print(e)

        print("完成")

    def generate_report(self):

        # 计算余弦相似度
        embedding_vector_list = []
        for paper_summary_str in self.knowledge_finding_config['paper_summary_str_list']:
            embedding_vector = APILLMParser.call_embedding_llm_api(
                str_list=[paper_summary_str]
            )
            embedding_vector_list.append(embedding_vector)

        # 计算两两相似度
        similarity_matrix = cosine_similarity(np.array(embedding_vector_list))

        # 获取相似度矩阵的上三角部分（避免重复计算）
        triu_indices = np.triu_indices_from(similarity_matrix, k=1)

        # 提取相似度值及其对应的向量索引
        similarities = similarity_matrix[triu_indices]
        pairs = list(zip(triu_indices[0], triu_indices[1], similarities))

        # 按相似度从高到低排序
        sorted_pairs = sorted(pairs, key=lambda x: x[2], reverse=True)

        # 将相似度矩阵转换为距离矩阵（1 - 相似度）
        distance_matrix = 1 - similarity_matrix



