from flask import Blueprint

from entity.common.Req import Req
from service.WebCrawlService import WebCrawlService

# 实例化/wc的Blueprint
wc_bp = Blueprint(
    name="wc",
    import_name=__name__,
    url_prefix="/wc"
)


# 开始爬取网页
@wc_bp.get("/getCrawlTaskOptions")
def get_crawl_task_options_api():

    return WebCrawlService.get_crawl_task_options()


# 开始爬取网页
@wc_bp.get("/getCrawlTaskParams")
def get_crawl_task_params_api():
    crawl_task = Req.receive_get_param("crawlTask")

    return WebCrawlService.get_crawl_task_params(crawl_task)


# 开始爬取网页
@wc_bp.get("/getCrawlTaskStepperData")
def get_crawl_task_stepper_data_api():
    crawl_task = Req.receive_get_param("crawlTask")

    return WebCrawlService.get_crawl_task_stepper_data(crawl_task)


# 开始爬取网页
@wc_bp.get("/getDefaultWebCrawlConfig")
def get_default_web_crawl_config_api():

    return WebCrawlService.get_default_web_crawl_config()


# 开始训练模型
@wc_bp.post("/startWebCrawlRunning")
def start_web_crawl_running_api():
    web_crawl_config = Req.receive_post_param("webCrawlConfig")
    progress_id = Req.receive_post_param("progressId")
    token = Req.receive_header_token()

    return WebCrawlService.start_web_crawl_running(
        web_crawl_config=web_crawl_config,
        progress_id=progress_id,
        token=token
    )


# 开始爬取网页
@wc_bp.get("/getCrawlSearchPaperInfoWebsiteOptions")
def get_crawl_search_paper_info_website_options_api():
    crawl_task = Req.receive_get_param("crawlTask")

    return WebCrawlService.get_crawl_search_paper_info_website_options(
        crawl_task=crawl_task
    )


# 开始爬取网页
@wc_bp.get("/getCrawlSearchPaperAbstractWebsiteOptions")
def get_crawl_search_paper_abstract_website_options_api():
    crawl_task = Req.receive_get_param("crawlTask")

    return WebCrawlService.get_crawl_search_paper_abstract_website_options(
        crawl_task=crawl_task
    )


# 开始爬取网页
@wc_bp.get("/getCrawlSearchPaperPDFWebsiteOptions")
def get_crawl_search_paper_PDF_website_options_api():
    crawl_task = Req.receive_get_param("crawlTask")

    return WebCrawlService.get_crawl_search_paper_PDF_website_options(
        crawl_task=crawl_task
    )


# 开始爬取网页
@wc_bp.get("/getCrawlMyTaskOptions")
def get_crawl_my_task_options_api():
    crawl_task = Req.receive_get_param("crawlTask")
    token = Req.receive_header_token()

    return WebCrawlService.get_crawl_my_task_options(
        crawl_task=crawl_task,
        token=token
    )


# 开始爬取网页
@wc_bp.get("/getCrawlGetPaperAISummaryOptions")
def get_crawl_get_paper_ai_summary_options_api():
    crawl_task = Req.receive_get_param("crawlTask")

    return WebCrawlService.get_crawl_get_paper_ai_summary_options(
        crawl_task=crawl_task
    )