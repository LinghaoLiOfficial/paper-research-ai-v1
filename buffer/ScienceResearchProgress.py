import copy

from buffer.CommonBuffer import CommonBuffer
from utils.common.TimeParser import TimeParser
import threading


class ScienceResearchProgress(CommonBuffer):

    DEFAULT_USER_PATTERN = {
            "log": [],
            "total_progress": {
                "now": 0,
                "all": 100
            },
            "current_stage": "",
            "current_task_id": "",
            "current_task_name": "",
        }

    _USER_DATA = {}

    @classmethod
    def get_all(cls):
        return cls._USER_DATA

    @classmethod
    def get(cls, user_id):
        if user_id in cls._USER_DATA.keys():
            return cls._USER_DATA[user_id]

        return cls.DEFAULT_USER_PATTERN

    @classmethod
    def create(cls, user_id, task_id, task_name):
        with cls._lock:
            cls._USER_DATA[user_id] = copy.deepcopy(cls.DEFAULT_USER_PATTERN)
            cls.update_current_task(
                user_id=user_id,
                task_id=task_id,
                task_name=task_name
            )

    @classmethod
    def update_log(cls, user_id, new_message):
        # 在列表的最前面插入
        cls._USER_DATA[user_id]["log"].insert(0, {
            "timestamp": str(TimeParser.get_current_time()),
            "message": new_message
        })

    @classmethod
    def update_total_progress(cls, user_id, adding):
        # 防止超过最大值
        if cls._USER_DATA[user_id]["total_progress"]["now"] + adding > cls._USER_DATA[user_id]["total_progress"]["all"]:
            cls._USER_DATA[user_id]["total_progress"]["now"] = cls._USER_DATA[user_id]["total_progress"]["all"]
        else:
            cls._USER_DATA[user_id]["total_progress"]["now"] += adding

    @classmethod
    def update_current_stage(cls, user_id, new_stage):
        cls._USER_DATA[user_id]["current_stage"] = new_stage

    @classmethod
    def update_current_task(cls, user_id, task_id, task_name):
        cls._USER_DATA[user_id]["current_task_id"] = task_id
        cls._USER_DATA[user_id]["current_task_name"] = task_name
