from entity.common.StrucResult import StrucResult
from utils.web_spider.WebSpider import WebSpider
import threading
from collections import deque


class WebCrawlBuffer:

    task_deque = deque()
    is_running = False
    global_info = {}

    @classmethod
    def check_running(cls):
        return cls.is_running

    @classmethod
    def start_running(cls, web_crawl_config):

        # 将当前任务添加到任务队列内
        cls.task_deque.append(web_crawl_config)

        if not cls.is_running:
            web_crawl_thread = threading.Thread(target=cls.thread_start_running)
            web_crawl_thread.start()

            return StrucResult.build_success(
                code=30001,
                message="爬虫任务队列为空，当前爬虫任务已开始"
            )

        return StrucResult.build_success(
                code=30002,
                message=f"爬虫任务队列内共有{len(cls.task_deque)}个任务，已将当前爬虫任务添加至队列末尾"
            )

    @classmethod
    def thread_start_running(cls):

        cls.is_running = True

        while len(cls.task_deque) > 0:
            web_spider = WebSpider()
            web_crawl_config = cls.task_deque.popleft()
            web_spider.start(web_crawl_config)

        cls.is_running = False
