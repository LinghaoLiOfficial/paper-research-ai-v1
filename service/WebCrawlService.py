import os

from buffer.WebCrawlBuffer import WebCrawlBuffer
from config.name.web_crawl.CrawlGetPaperAISummaryEnName import CrawlGetPaperAISummaryEnName
from config.name.web_crawl.CrawlGetPaperAISummaryZhName import CrawlGetPaperAISummaryZhName
from config.name.web_crawl.CrawlSearchPaperAbstractWebsiteEnName import CrawlSearchPaperAbstractWebsiteEnName
from config.name.web_crawl.CrawlSearchPaperAbstractWebsiteZhName import CrawlSearchPaperAbstractWebsiteZhName
from config.name.web_crawl.CrawlSearchPaperInfoWebsiteEnName import CrawlSearchPaperInfoWebsiteEnName
from config.name.web_crawl.CrawlSearchPaperInfoWebsiteZhName import CrawlSearchPaperInfoWebsiteZhName
from config.name.web_crawl.CrawlSearchPaperPDFWebsiteEnName import CrawlSearchPaperPDFWebsiteEnName
from config.name.web_crawl.CrawlSearchPaperPDFWebsiteZhName import CrawlSearchPaperPDFWebsiteZhName
from config.name.web_crawl.CrawlSearchPaperSubTaskEnName import CrawlSearchPaperSubTaskEnName
from config.name.web_crawl.CrawlTaskEnName import CrawlTaskEnName
from config.name.web_crawl.CrawlTaskParamDefault import CrawlTaskParamDefault
from config.name.web_crawl.CrawlTaskParamEnName import CrawlTaskParamEnName
from config.name.web_crawl.CrawlTaskParamZhName import CrawlTaskParamZhName
from config.name.web_crawl.CrawlTaskZhName import CrawlTaskZhName
from entity.common.Resp import Resp
from mapper.WebCrawlMapper import WebCrawlMapper
from utils.common.JWTParser import JWTParser


class WebCrawlService:

    CRAWL_CSV_PATH = "./storage/{}/web_crawl/{}"

    @classmethod
    def get_crawl_my_task_options(cls, crawl_task, token):

        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        web_crawl_mapper_result = WebCrawlMapper.select_progress_where_owner(
            owner=user_id
        )
        if not web_crawl_mapper_result.check:
            return Resp.build_db_error()

        crawl_my_task_options = []
        for i, progress in enumerate(web_crawl_mapper_result.get_data_on_results()):
            crawl_my_task_options.append({
                "id": i,
                "label": progress.progress_id,
                "name": progress.progress_id
            })

        return Resp.build_success(data={
            "crawlMyTaskOptions": crawl_my_task_options
        })

    @classmethod
    def get_crawl_get_paper_ai_summary_options(cls, crawl_task):
        crawl_get_paper_ai_summary_options = {
            CrawlTaskEnName.THEME_PAPER_RELATION_MINING: [
                {
                    "id": 0,
                    "label": CrawlGetPaperAISummaryZhName.DEEPSEEK,
                    "name": CrawlGetPaperAISummaryEnName.DEEPSEEK
                },
            ]
        }.get(crawl_task)

        return Resp.build_success(data={
            "crawlGetPaperAISummaryOptions": crawl_get_paper_ai_summary_options
        })

    @classmethod
    def get_crawl_search_paper_PDF_website_options(cls, crawl_task):
        crawl_search_paper_pdf_website_options = {
            CrawlTaskEnName.THEME_PAPER_RELATION_MINING: [
                {
                    "id": 0,
                    "label": CrawlSearchPaperPDFWebsiteZhName.ENDNOTE_ORIGINAL,
                    "name": CrawlSearchPaperPDFWebsiteEnName.ENDNOTE_ORIGINAL
                },
                {
                    "id": 1,
                    "label": CrawlSearchPaperPDFWebsiteZhName.SCIHUB,
                    "name": CrawlSearchPaperPDFWebsiteEnName.SCIHUB
                },
            ]
        }.get(crawl_task)

        return Resp.build_success(data={
            "crawlSearchPaperPDFWebsiteOptions": crawl_search_paper_pdf_website_options
        })

    @classmethod
    def get_crawl_search_paper_abstract_website_options(cls, crawl_task):
        crawl_search_paper_abstract_website_options = {
            CrawlTaskEnName.THEME_PAPER_RELATION_MINING: [
                {
                    "id": 0,
                    "label": CrawlSearchPaperAbstractWebsiteZhName.GOOGLE_SCHOLAR,
                    "name": CrawlSearchPaperAbstractWebsiteEnName.GOOGLE_SCHOLAR
                }
            ]
        }.get(crawl_task)

        return Resp.build_success(data={
            "crawlSearchPaperAbstractWebsiteOptions": crawl_search_paper_abstract_website_options
        })

    @classmethod
    def get_crawl_search_paper_info_website_options(cls, crawl_task):
        crawl_search_paper_info_website_options = {
            CrawlTaskEnName.THEME_PAPER_RELATION_MINING: [
                {
                    "id": 0,
                    "label": CrawlSearchPaperInfoWebsiteZhName.ZJUT_LIB,
                    "name": CrawlSearchPaperInfoWebsiteEnName.ZJUT_LIB
                },
                {
                    "id": 1,
                    "label": CrawlSearchPaperInfoWebsiteZhName.WEB_OF_SCIENCE,
                    "name": CrawlSearchPaperInfoWebsiteEnName.WEB_OF_SCIENCE
                },
            ]
        }.get(crawl_task)

        return Resp.build_success(data={
            "crawlSearchPaperInfoWebsiteOptions": crawl_search_paper_info_website_options
        })

    @classmethod
    def get_crawl_task_options(cls):
        crawl_task_options = [
            {
                "id": 0,
                "label": CrawlTaskZhName.THEME_PAPER_RELATION_MINING,
                "name": CrawlTaskEnName.THEME_PAPER_RELATION_MINING
            },
            {
                "id": 1,
                "label": CrawlTaskZhName.OPINION_TOPIC_HEAT_QUANTIFY,
                "name": CrawlTaskEnName.OPINION_TOPIC_HEAT_QUANTIFY
            },
        ]

        return Resp.build_success(data={
            "crawlTaskOptions": crawl_task_options
        })

    @classmethod
    def get_crawl_task_params(cls, crawl_task):
        crawl_task_params = {
            CrawlTaskEnName.THEME_PAPER_RELATION_MINING: {
                "input": [
                    {
                        "id": 0,
                        "label": CrawlTaskParamZhName.SEARCH_WORDS,
                        "name": CrawlTaskParamEnName.SEARCH_WORDS,
                    },
                    {
                        "id": 1,
                        "label": CrawlTaskParamZhName.MAX_NUM,
                        "name": CrawlTaskParamEnName.MAX_NUM
                    },
                ],
                "checkboxgroup": [
                    {
                        "id": 0,
                        "label": CrawlTaskParamZhName.SUBJECT_FILTER_ENGINEERING,
                        "name": CrawlTaskParamEnName.SUBJECT_FILTER_ENGINEERING,
                    },
                    {
                        "id": 1,
                        "label": CrawlTaskParamZhName.SUBJECT_FILTER_APPLIED_SCIENCE,
                        "name": CrawlTaskParamEnName.SUBJECT_FILTER_APPLIED_SCIENCE,
                    },
                    {
                        "id": 2,
                        "label": CrawlTaskParamZhName.SUBJECT_FILTER_BUSINESS,
                        "name": CrawlTaskParamEnName.SUBJECT_FILTER_BUSINESS,
                    },
                    {
                        "id": 3,
                        "label": CrawlTaskParamZhName.SUBJECT_FILTER_ECONOMICS,
                        "name": CrawlTaskParamEnName.SUBJECT_FILTER_ECONOMICS,
                    },
                ],
            }
        }.get(crawl_task)

        return Resp.build_success(data={
            "crawlTaskParams": crawl_task_params
        })

    @classmethod
    def get_crawl_task_stepper_data(cls, crawl_task):
        crawl_task_stepper_data = {
            CrawlTaskEnName.THEME_PAPER_RELATION_MINING: [
                {
                    "id": 0,
                    "name": "large",
                    "value": 0
                },
                {
                    "id": 1,
                    "name": "small",
                    "value": 0
                },
                {
                    "id": 2,
                    "name": "large",
                    "value": 0
                },
                {
                    "id": 3,
                    "name": "small",
                    "value": 0
                },
                {
                    "id": 4,
                    "name": "large",
                    "value": 0
                },
                {
                    "id": 5,
                    "name": "small",
                    "value": 0
                },
                {
                    "id": 6,
                    "name": "large",
                    "value": 0
                },
                {
                    "id": 7,
                    "name": "small",
                    "value": 0
                },
            ]
        }.get(crawl_task)

        return Resp.build_success(data={
            "crawlTaskStepperData": crawl_task_stepper_data
        })

    @classmethod
    def get_default_web_crawl_config(cls):

        default_web_crawl_config = {
            "taskManage": {
                "ChooseCrawlTaskDropdown": CrawlTaskEnName.THEME_PAPER_RELATION_MINING,
                "CrawlTaskParamsInputs": {
                    CrawlTaskParamEnName.SEARCH_WORDS: CrawlTaskParamDefault.SEARCH_WORDS,
                    CrawlTaskParamEnName.MAX_NUM: CrawlTaskParamDefault.MAX_NUM,
                },
                "CrawlTaskParamsCheckboxGroup": {
                    CrawlTaskParamEnName.SUBJECT_FILTER_ENGINEERING: CrawlTaskParamDefault.SUBJECT_FILTER_ENGINEERING,
                    CrawlTaskParamEnName.SUBJECT_FILTER_APPLIED_SCIENCE: CrawlTaskParamDefault.SUBJECT_FILTER_APPLIED_SCIENCE,
                    CrawlTaskParamEnName.SUBJECT_FILTER_ECONOMICS: CrawlTaskParamDefault.SUBJECT_FILTER_ECONOMICS,
                    CrawlTaskParamEnName.SUBJECT_FILTER_BUSINESS: CrawlTaskParamDefault.SUBJECT_FILTER_BUSINESS,
                },

            }
        }

        return Resp.build_success(data={
            "defaultWebCrawlConfig": default_web_crawl_config
        })

    @classmethod
    def start_web_crawl_running(cls, web_crawl_config, progress_id, token):

        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        # 创建文件夹存储文件
        base_path = cls.CRAWL_CSV_PATH.format(user_id, progress_id)
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        web_crawl_mapper_result = WebCrawlMapper.select_progress_where_id(
            id=progress_id
        )
        if not web_crawl_mapper_result.check:
            return Resp.build_db_error()

        progress_obj_list = web_crawl_mapper_result.get_data_on_results()

        if len(progress_obj_list) > 0:
            # 已找到
            progress_obj = progress_obj_list[0]
        else:
            # 未找到
            web_crawl_mapper_result_1 = WebCrawlMapper.insert_progress(
                id=progress_id,
                owner=user_id
            )
            if not web_crawl_mapper_result_1.check:
                return Resp.build_db_error()

            web_crawl_mapper_result_2 = WebCrawlMapper.select_progress_where_id(
                id=progress_id
            )
            if not web_crawl_mapper_result_2.check:
                return Resp.build_db_error()

            progress_obj = web_crawl_mapper_result_2.get_data_on_results()[0]

        progress_dict = {
            "progress_step_1": progress_obj.progress_step_1,
            "progress_step_2": progress_obj.progress_step_2,
            "progress_step_3": progress_obj.progress_step_3,
            "progress_step_4": progress_obj.progress_step_3,
        }
        web_crawl_config["progress"] = progress_dict

        file_path_dict = {
            "path_step_1": f"{base_path}/{CrawlSearchPaperSubTaskEnName.SEARCH_PAPER_INFO}",
            "path_step_2": f"{base_path}/{CrawlSearchPaperSubTaskEnName.SEARCH_PAPER_ABSTRACT}",
            "path_step_3": f"{base_path}/{CrawlSearchPaperSubTaskEnName.SEARCH_PAPER_PDF}",
            "path_step_4": f"{base_path}/{CrawlSearchPaperSubTaskEnName.GET_PAPER_AI_SUMMARY}",
        }
        web_crawl_config["file_path"] = file_path_dict

        web_crawl_config["progress_id"] = progress_id

        web_crawl_buffer_result = WebCrawlBuffer.start_running(web_crawl_config)

        return Resp.build_success(
            code=web_crawl_buffer_result.code,
            message=web_crawl_buffer_result.message
        )
