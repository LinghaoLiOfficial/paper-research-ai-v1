from entity.common.Resp import Resp
from utils.visual_simulation.VisualSimulationTask import VisualSimulationTask


class VisualSimulationService:

    @classmethod
    def restart_simulation(cls, model, token):

        task_result = VisualSimulationTask.restart()

        return Resp.build_success()
