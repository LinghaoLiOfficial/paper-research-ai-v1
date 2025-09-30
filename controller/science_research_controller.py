from flask import Blueprint

from entity.common.Req import Req
from service.ScienceResearchService import ScienceResearchService

# 实例化Blueprint
sr_bp = Blueprint(
    name="sr",
    import_name=__name__,
    url_prefix="/sr"
)


# 获取时间演化趋势图
@sr_bp.get("/getTimeEvolveTrend")
def get_time_evolve_trend_api():
    task_id = Req.receive_get_param("taskId")
    content_perspective = Req.receive_get_param("contentPerspective")

    return ScienceResearchService.get_time_evolve_trend(
        task_id=task_id,
        content_perspective=content_perspective
    )


# 获取知识图谱的总结
@sr_bp.get("/getKnowledgeGraphSummary")
def get_knowledge_graph_summary_api():
    task_id = Req.receive_get_param("taskId")
    content_perspective = Req.receive_get_param("contentPerspective")

    return ScienceResearchService.get_knowledge_graph_summary(
        task_id=task_id,
        content_perspective=content_perspective
    )


# 获取用户的知识网络
@sr_bp.get("/getKnowledgeNetwork")
def get_knowledge_network_api():
    task_id = Req.receive_get_param("taskId")
    content_perspective = Req.receive_get_param("contentPerspective")

    return ScienceResearchService.get_knowledge_network(
        task_id=task_id,
        content_perspective=content_perspective
    )


# 获取知识网络的节点信息
@sr_bp.get("/getKnowledgeGraphNodeDetailInfo")
def get_knowledge_graph_node_detail_info_api():
    token = Req.receive_header_token()
    node_id = Req.receive_get_param("nodeId")

    return ScienceResearchService.get_knowledge_graph_node_detail_info(
        node_id=node_id,
        token=token
    )


# 获取知识网络边结构选项
@sr_bp.get("/getContentPerspectiveDropdown")
def get_content_perspective_dropdown_api():
    return ScienceResearchService.get_content_perspective_dropdown()


# 获取用户所有的任务
@sr_bp.get("/getMySpaceTasks")
def get_my_space_tasks_api():
    token = Req.receive_header_token()

    return ScienceResearchService.get_my_space_tasks(
        token=token
    )


# 开始构建论文知识网络
@sr_bp.post("/startCreateKnowledgeGraph")
def start_create_knowledge_graph_api():
    task_params = Req.receive_post_param("taskParams")
    task_type = Req.receive_post_param("taskType")
    token = Req.receive_header_token()

    return ScienceResearchService.start_create_knowledge_graph(
        task_params=task_params,
        task_type=task_type,
        token=token
    )


# 删除知识网络任务
@sr_bp.post("/taskDeleteTask")
def task_delete_task_api():
    task_id = Req.receive_post_param("taskId")
    token = Req.receive_header_token()

    return ScienceResearchService.task_delete_task(
        task_id=task_id,
        token=token
    )


# 重试知识网络任务
@sr_bp.post("/taskRetryTask")
def task_retry_task_api():
    task_id = Req.receive_post_param("taskId")
    token = Req.receive_header_token()

    return ScienceResearchService.task_retry_task(
        task_id=task_id,
        token=token
    )


# 获取用户的当前任务进度
@sr_bp.get("/getTaskProgress")
def get_task_progress_api():
    token = Req.receive_header_token()

    return ScienceResearchService.get_task_progress(
        token=token
    )

