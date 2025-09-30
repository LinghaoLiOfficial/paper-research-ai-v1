import threading
from concurrent.futures import ThreadPoolExecutor


class CommonBuffer:
    _lock = threading.Lock()  # 类级锁，保护共享资源
    _thread_pool = ThreadPoolExecutor(max_workers=1)  # 线程池

    @classmethod
    def shutdown(cls):
        with cls._lock:
            # 安全关闭线程池
            cls._thread_pool.shutdown(wait=False)

    @classmethod
    def clear_and_reinit(cls):
        with cls._lock:
            # 停止现有线程池
            cls._thread_pool.shutdown(wait=False)  # 不等待未完成任务

            # 重新初始化线程池
            cls._thread_pool = ThreadPoolExecutor(max_workers=1)

