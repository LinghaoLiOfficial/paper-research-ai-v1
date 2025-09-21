from buffer.CommonBuffer import CommonBuffer
from entity.common.StrucResult import StrucResult
from utils.data_analysis.ModelTrainingTask import ModelTrainingTask


class ModelTrainingBuffer(CommonBuffer):

    @classmethod
    def start_running(cls, config):
        with cls._lock:
            # 将当前任务添加到任务队列内
            cls._thread_pool.submit(cls._process_task, config)

            if cls._thread_pool._work_queue.qsize() == 0:
                return StrucResult.build_success(
                    code=30001,
                    message="模型训练任务队列为空，当前任务已开始"
                )
            return StrucResult.build_success(
                    code=30002,
                    message=f"当前队列内共有{cls._thread_pool._work_queue.qsize()}个任务，已将任务添加至队列末尾，等待中"
                )

    @classmethod
    def _process_task(cls, config):
        # 执行任务（不持有锁，允许其他线程操作队列）
        try:
            data_analysis = ModelTrainingTask(data_analysis_config=config)
            data_analysis.start()
        except Exception as e:
            print(f"数据分析任务执行失败: {e}")


