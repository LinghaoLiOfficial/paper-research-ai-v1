from buffer.CommonBuffer import CommonBuffer
from entity.common.StrucResult import StrucResult
from mapper.ScienceResearchMapper import ScienceResearchMapper
from buffer.ScienceResearchProgress import ScienceResearchProgress
from utils.common.EmailSender import EmailSender


class KnowledgeGraphTaskBuffer(CommonBuffer):

    @classmethod
    def start_running(cls, method, info):
        with cls._lock:
            # 将当前任务添加到任务队列内
            cls._thread_pool.submit(cls._process_task, method, info)

            if cls._thread_pool._work_queue.qsize() == 0:
                return StrucResult.build_success(
                    code=30001,
                    message="任务队列为空，当前任务已开始"
                )
            return StrucResult.build_success(
                    code=30002,
                    message=f"当前队列内共有{cls._thread_pool._work_queue.qsize()}个任务，已将任务添加至队列末尾，等待中"
                )

    @classmethod
    def _process_task(cls, method, info):
        # 执行任务（不持有锁，允许其他线程操作队列）
        try:
            method()

            # 修改任务状态
            for search_id in info['search_id_list']:
                mysql_result = ScienceResearchMapper.update_paper_search_set_task_status({
                    "task_status": "completed",
                    "search_id": search_id
                })

        except Exception as e:
            ScienceResearchProgress.update_log(user_id=info['user_id'], new_message=f"!任务执行失败: {e}")

            # 修改任务状态
            for search_id in info['search_id_list']:
                mysql_result = ScienceResearchMapper.update_paper_search_set_task_status({
                    "task_status": "failed",
                    "search_id": search_id
                })

            mysql_result1 = ScienceResearchMapper.select_user_where_user_id({
                "user_id": info['user_id']
            })

            user_email = mysql_result1.get_data_on_results()[0]['user_email']

            # 发送邮件
            try:
                EmailSender.send_knowledge_graph_task_error(
                    email=user_email,
                    task_id=info['task_id'],
                    task_name=info['task_name']
                )
            except Exception as e:
                print(e)


