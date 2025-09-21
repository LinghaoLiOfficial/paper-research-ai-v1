from flask import Blueprint

from entity.common.Req import Req
from service.KnowledgeBaseService import KnowledgeBaseService
from service.VisualSimulationService import VisualSimulationService

# 实例化Blueprint
vs_bp = Blueprint(
    name="vs",
    import_name=__name__,
    url_prefix="/vs"
)


# 创建文件
@vs_bp.post("/restartSimulation")
def restart_simulation_api():
    model = Req.receive_post_param("model")
    token = Req.receive_header_token()

    return VisualSimulationService.restart_simulation(
        model=model,
        token=token
    )
