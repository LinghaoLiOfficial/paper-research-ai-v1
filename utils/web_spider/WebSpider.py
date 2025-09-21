from config.name.web_crawl.CrawlTaskParamEnName import CrawlTaskParamEnName
from config.name.web_crawl.CrawlTaskParamZhName import CrawlTaskParamZhName
from mapper.WebCrawlMapper import WebCrawlMapper
from utils.web_spider.WebCrawlParser import WebCrawlParser


class WebSpider:

    def __init__(self):
        self.web_crawl_config = None

    def start(self, web_crawl_config):
        self.web_crawl_config = web_crawl_config

        crawl_task_param_name_dict = dict(zip(CrawlTaskParamEnName.to_list(), CrawlTaskParamZhName.to_list()))

        # 根据多个关键词检索论文标题、时间、作者、期刊
        if self.web_crawl_config['progress']['progress_step_1'] == 0:

            result = WebCrawlParser.crawl_papers_info(
                target_url="https://www.lib.zjut.edu.cn",
                search_words=self.web_crawl_config['taskManage']['CrawlTaskParamsInputs']['search_words'],
                search_box_xpath=".//input[@id='ipt1']",
                search_button_xpath=".//div[@id='btn2']",
                mark_xpath=".//div[@class='documentSummary document-grid']",
                checkbox_xpath_list=[
                    f".//input[@aria-label='{crawl_task_param_name_dict[k]}']" for k, v in self.web_crawl_config['taskManage']['CrawlTaskParamsCheckboxGroup'].items() if v == 1
                ],
                content_xpath_dict={
                    "title": ".//a[@class='ng-binding ng-scope ellipsis-3']",
                    "time": ".//div[@class='shortSummary ng-scope']//span[@class='ng-binding ng-scope']",
                    "authors": ".//div[@label='著者']//a[@class='customPrimaryLink ng-binding']",
                    "journal": ".//div[@class='shortSummary ng-scope']//a[@class='customPrimaryLink ng-binding']"
                },
                page_xpath=".//a[@aria-label='Page {}']",
                page_bar_xpath=".//div[@view='search']",
                pause_function=True,
                max_num=self.web_crawl_config['taskManage']['CrawlTaskParamsInputs']['max_num'],
                file_path=self.web_crawl_config['file_path']['path_step_1']
            )

            if result:
                web_crawl_mapper_result = WebCrawlMapper.update_progress_set_step("progress_step_1", 1, self.web_crawl_config['progress_id'])
                if not web_crawl_mapper_result.check:
                    return False

        # # 根据表格中的论文标题检索论文摘要
        # if self.web_crawl_config['progress']['progress_step_2'] == 0:
        #
        #     WebCrawlParser.crawl_papers_abstract_by_scholar_engine(
        #         target_url="https://ac.zhike.in",
        #         search_box_xpath=".//div[@class='mk-side-form']//input[@id='url-input']",
        #         search_button_xpath=".//div[@class='mk-side-form']//button[@id='download-button']",
        #         pause_function=True,
        #         file_path=self.web_crawl_config['file_path']['path_step_2'],
        #         last_file_path=self.web_crawl_config['file_path']['path_step_1']
        #     )

        # 根据表格中的论文标题检索论文原文PDF
        if self.web_crawl_config['progress']['progress_step_3'] == 0:
            WebCrawlParser.crawl_papers_pdf_original_by_scihub(
                target_url="https://www.sci-hub.st",
                # search_box_xpath=".//div[contains(@class, 'form-group')]//input[@id='search-input']",
                search_box_xpath=".//textarea[@name='request']",
                # search_button_xpath=".//div[contains(@class, 'form-group')]//button[@id='search-button']",
                search_button_xpath=".//button[@type='submit']",
                pause_function=True,
                file_path=self.web_crawl_config['file_path']['path_step_3'],
                last_file_path=self.web_crawl_config['file_path']['path_step_1']
            )

        # # 根据表格中的论文标题检索论文原文PDF
        # if self.web_crawl_config['progress']['progress_step_3'] == 0:
        #     WebCrawlParser.crawl_papers_pdf_original_by_scholar_engine(
        #         target_url="https://sc.panda985.com",
        #         search_box_xpath=".//div[contains(@class, 'form-group')]//input[@id='search-input']",
        #         search_button_xpath=".//div[contains(@class, 'form-group')]//button[@id='search-button']",
        #         pause_function=True,
        #         file_path=self.web_crawl_config['file_path']['path_step_3'],
        #         last_file_path=self.web_crawl_config['file_path']['path_step_1']
        #     )

        # 根据论文原文PDF获取AI总结研究问题、研究方法、贡献、局限性
        if self.web_crawl_config['progress']['progress_step_4'] == 0:
            WebCrawlParser.crawl_summary_from_llm_web(
                target_url="https://chat.deepseek.com/",
                search_box_xpath=".//textarea[@id='chat-input']",
                search_button_xpath=".//div[@class='f286936b']",
                pause_function=True
            )
