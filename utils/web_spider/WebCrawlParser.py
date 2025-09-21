import copy
import os
import requests
import fake_useragent
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
import pandas as pd
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.proxy import Proxy, ProxyType
from algorithm.DistanceCalculation import DistanceCalculation
from config.name.web_crawl.CrawlSearchPaperSubTaskEnName import CrawlSearchPaperSubTaskEnName
from utils.common.DelayParser import DelayParser
from utils.common.InputListener import InputListener
import subprocess
import json


class WebCrawlParser:

    @classmethod
    def configure(cls, is_random_user_agent: bool, is_headless: bool, web_url: str = None):

        # 使用Edge浏览器来模拟
        options = Options()
        # 启用Chromium特性
        options.use_chromium = True

        """
            配置浏览器的选项
        """

        # 设置User-Agent
        if is_random_user_agent:
            random_user_agent = fake_useragent.UserAgent().random
            options.add_argument(f"user-agent={random_user_agent}")
        else:
            options.add_argument(f"user-agent={os.getenv('LOCAL_USER_AGENT')}")
        # 隐藏自动化特征
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        options.add_argument("--start-maximized")  # 启动浏览器时最大化窗口
        options.add_argument("window-size=1920x1080")  # 设置窗口分辨率
        if is_headless:
            options.add_argument("--headless")  # 如果希望隐藏浏览器窗口可以启用此行

        #TODO
        # download_dir = ""
        # prefs = {
        #     "download.default_directory": download_dir,  # 设置默认下载路径
        #     "download.prompt_for_download": False,  # 禁用下载提示
        #     "download.directory_upgrade": True,  # 启用下载目录升级
        #     "safebrowsing.enabled": True  # 启用安全浏览
        # }
        # options.add_experimental_option("prefs", prefs)

        # 是否使用vpn代理
        proxy = False
        if web_url:
            vpn_using_sites = ["https://www.google.com/"]
            for site in vpn_using_sites:
                if site in web_url:
                    proxy = True
                    break
        if not proxy:
            options.add_argument('--no-proxy-server')  # 禁用代理

        # 远程接管用户浏览器
        will_takeover = True
        if will_takeover:
            # 加载用户数据目录
            user_data_dir = "D:\\Users\\Z\\PycharmProjects\\MyBlogFlask\\external\\web_crawl\\user_data"
            options.add_argument(f"--user-data-dir={user_data_dir}")
            options.add_argument("--remote-debugging-port=9222")

        # TODO: 可自动更新网络驱动器（自动下载）
        # 使用本地的Edge WebDriver
        service = Service(os.getenv("EDGE_DRIVER_PATH"))
        driver = webdriver.Edge(service=service, options=options)

        # 使用 execute_cdp_cmd() 在页面加载前修改 navigator.webdriver 属性，将其设置为 undefined，从而避免被检测到。
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": """
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
            """
        })

        return driver

    @classmethod
    def switch_to_latest_window(cls, driver):
        # 获取所有窗口句柄
        window_handles = driver.window_handles
        # 切换到最新打开的窗口
        driver.switch_to.window(window_handles[-1])
        if driver.current_url == 'chrome-extension://hbfhdcdeikaehendoojplfpjcfifnlme/offscreen.html':
            driver.switch_to.window(window_handles[-2])

        return len(window_handles)

    @classmethod
    def wait_until_element(cls, driver, element: str, max_wait_time: int = 30):
        # 等待特定元素加载完成
        item = WebDriverWait(driver, max_wait_time).until(EC.presence_of_element_located((By.XPATH, element)))

        return item

    # 滚动到页面底部
    @classmethod
    def scroll_to_bottom(cls, driver):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    # 根据元素强制点击
    @classmethod
    def direct_click_element(cls, driver, element):
        driver.execute_script("arguments[0].click();", element)

    @classmethod
    def crawl_papers_info(cls, target_url: str, search_words: str, search_box_xpath: str, search_button_xpath: str,
                          mark_xpath: str, checkbox_xpath_list: list, content_xpath_dict: dict, page_xpath: str,
                          page_bar_xpath: str, pause_function: bool = False, max_num: int = None, file_path=None):

        flag = True

        if pause_function:
            InputListener.activate()

        driver = cls.configure(
            is_random_user_agent=False,
            is_headless=False,
            web_url=target_url
        )

        try:
            # 打开网站
            driver.get(target_url)
            DelayParser.execute_short_delay()

            InputListener.listen()

            # 搜索框输入
            search_box = driver.find_element(By.XPATH, search_box_xpath)
            search_box.send_keys(search_words)
            DelayParser.execute_short_delay()

            # 点击搜索按钮
            search_button = driver.find_element(By.XPATH, search_button_xpath)
            search_button.click()
            DelayParser.execute_medium_delay()

            InputListener.listen()

            cls.switch_to_latest_window(driver)

            cls.wait_until_element(driver, mark_xpath)

            # 勾选条件
            for xpath in checkbox_xpath_list:
                checkbox = driver.find_element(By.XPATH, xpath)
                cls.direct_click_element(driver, checkbox)
                DelayParser.execute_medium_delay()

            InputListener.listen()

            # 读取筛选文件，转变为列表
            df = pd.read_excel(os.getenv("ABS_JOURNAL_XLSX_PATH"), sheet_name=0)
            filter_list = df.loc[:, "title"].tolist()

            # 分页数据爬取
            sections_list = []
            page = 2
            continue_to_new_page = True
            while continue_to_new_page:

                cls.scroll_to_bottom(driver)

                # 获取所有搜索结果
                search_results = driver.find_elements(By.XPATH, mark_xpath)
                DelayParser.execute_long_delay()

                for j, result in enumerate(search_results):

                    title_target = result.find_elements(By.XPATH, content_xpath_dict["title"])
                    title = title_target[0].text.strip() if len(title_target) > 0 else ""

                    time_list_target = result.find_elements(By.XPATH, content_xpath_dict["time"])
                    time_list = [y for y in time_list_target[0].text.strip().split(",") if y not in ["", " "]] if len(
                        time_list_target) > 0 else []

                    publish_time = time_list[0] if len(time_list) >= 1 else ""

                    authors_target = result.find_elements(By.XPATH, content_xpath_dict["authors"])
                    authors = [x.text.strip().replace("……", "").replace("...", "").replace("更多", "more...") for x in
                               authors_target] if len(authors_target) > 0 else []

                    journal_target = result.find_elements(By.XPATH, content_xpath_dict["journal"])
                    journal = journal_target[0].text.strip() if len(journal_target) > 0 else ""

                    # if title not in ["", " ", None]:
                    #     for qualified in filter_list:
                    #         if CalculateDistance.calculate_levenshtein_distance(qualified.lower(), journal.lower()) >= 70:
                    #             sections_list.append({
                    #                 "title": title,
                    #                 "time": publish_time,
                    #                 "authors": authors,
                    #                 "journal": journal
                    #             })
                    #             break

                    if title != "" and publish_time != "" and authors != [] and journal != "":
                        if title not in ["", " ", None]:
                            sections_list.append({
                                "title": title,
                                "time": publish_time,
                                "authors": authors,
                                "journal": journal
                            })

                print(f"Page {page - 1} 已完成检索")
                print(f"目前已找到 {len(sections_list)} 个")

                # TODO: 特殊筛选条件
                # sections_list = [x for x in sections_list if not ("Part B" in x["journal"] or "Part C" in x["journal"])]

                # 防止超过最大数量
                if max_num and len(sections_list) >= max_num:
                    break

                cls.scroll_to_bottom(driver)

                page_bar = driver.find_element(By.XPATH, page_bar_xpath)
                DelayParser.execute_short_delay()

                # 翻页
                try:
                    a = WebDriverWait(page_bar, 5).until(
                        EC.presence_of_element_located((By.XPATH, page_xpath.replace("{}", str(page))))
                    )
                except TimeoutException:
                    print("已到达最后一页")
                    print(f"输出总数量: {len(sections_list)}")
                    continue_to_new_page = False
                    break
                cls.direct_click_element(driver, a)
                page += 1
                DelayParser.execute_medium_delay()

                InputListener.listen()

                cls.switch_to_latest_window(driver)

                cls.wait_until_element(driver, mark_xpath)

            # 截断
            if max_num:
                sections_list = sections_list[:max_num]

            # 保存到CSV文件
            df = pd.DataFrame(sections_list, columns=["title", "time", "authors", "journal"])
            df.to_csv(f"{file_path}.csv", index=False, encoding="utf-8-sig")

            print("爬虫程序正常结束")

        except Exception as e:
            print(f"[WebSpiderParser/crawl_papers_info] 爬虫抛出错误: {e}")

            flag = False

        finally:

            if pause_function:
                InputListener.deactivate()

            # 关闭浏览器
            driver.quit()

            return True if flag else False

    @classmethod
    def crawl_papers_abstract_by_scholar_engine(cls, target_url: str, search_box_xpath: str, search_button_xpath: str,
                                                pause_function: bool = False, file_path=None, last_file_path=None):

        if pause_function:
            InputListener.activate()

        driver = cls.configure(
            is_random_user_agent=False,
            is_headless=False
        )

        abstract_list = []
        try:
            # 打开网站
            driver.get(target_url)
            DelayParser.execute_short_delay()

            InputListener.listen()

            # 读取数据源文件，转变为列表
            df = pd.read_csv(last_file_path)
            paper_title_list = df.loc[:, "title"].tolist()

            # TODO
            # 读取摘要缓存
            if os.path.exists("./cache_abstract.csv"):
                old_cache_abstract = pd.read_csv("./cache_abstract.csv", index_col=0)
                abstract_list.extend(old_cache_abstract.loc[:, "abstract"].tolist())
                paper_title_list = paper_title_list[len(old_cache_abstract):]

            new_df = copy.deepcopy(df)
            for i, search_title in enumerate(paper_title_list):

                # 搜索框输入
                search_box = driver.find_element(By.XPATH, search_box_xpath)
                # 清空输入框内容
                search_box.clear()
                search_box.send_keys(search_title)
                DelayParser.execute_short_delay()

                # 点击搜索按钮
                search_button = driver.find_element(By.XPATH, search_button_xpath)
                search_button.click()
                DelayParser.execute_short_delay()

                cls.switch_to_latest_window(driver)

                InputListener.listen()

                DelayParser.execute_medium_delay()

                # 进入第一个网页链接
                try:
                    div = WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.XPATH, ".//div[@class='gs_fma_snp']"))
                    )

                    abstract_list.append(div.text.strip())
                    print(f"{i}: {search_title} 获取摘要成功")

                except Exception:
                    # 无法找到当前论文所在网页，跳过
                    abstract_list.append("Not found!")
                    print(f"{i}: {search_title} 获取摘要失败")

                # $x(".//div[@class='gs_fma gs_fma_p gs_invis']")
                # $x(".//div[@class='gs_fma_snp']")

                if len(driver.window_handles) == 2:
                    driver.close()
                    DelayParser.execute_short_delay()

                    cls.switch_to_latest_window(driver)
                    DelayParser.execute_short_delay()

            new_df["abstract"] = abstract_list
            # 保存到CSV文件
            new_df.to_csv("./abstract_papers.csv", index=True, encoding="utf-8-sig")

            print("爬虫程序正常结束")

        except Exception as e:
            print(f"[WebSpiderParser/crawl_papers_pdf_original] 爬虫抛出错误: {e}")

            # 保存已获取的缓存文本内容
            # 保存到CSV文件
            cache_df = pd.DataFrame({"abstract": abstract_list}, columns=["abstract"])
            cache_df.to_csv(file_path, index=True, encoding="utf-8-sig")

        finally:

            if pause_function:
                InputListener.deactivate()

            # 关闭浏览器
            driver.quit()

    @classmethod
    def crawl_papers_pdf_original_by_scihub(cls, target_url: str, search_box_xpath: str, search_button_xpath: str,
                                            pause_function: bool = False, file_path=None, last_file_path=None):

        if pause_function:
            InputListener.activate()

        driver = cls.configure(
            is_random_user_agent=False,
            is_headless=False
        )

        try:
            # 打开网站
            driver.get(target_url)
            DelayParser.execute_medium_delay()

            # “我是人类”界面
            try:
                div = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located(
                        (By.XPATH, ".//div[@id='label' and text()='我是人类']"))
                )

                # TODO:强制暂停程序

            except TimeoutException:
                pass

            InputListener.listen()

            # 读取数据源文件，转变为列表
            df = pd.read_csv(f"{last_file_path}.csv", index_col=False)
            paper_title_list = df.loc[:, "title"].tolist()

            for i, search_title in enumerate(paper_title_list):

                # 搜索框输入
                search_box = driver.find_element(By.XPATH, search_box_xpath)
                search_box.send_keys(search_title)
                DelayParser.execute_short_delay()

                # 点击搜索按钮
                search_button = driver.find_element(By.XPATH, search_button_xpath)
                search_button.click()
                DelayParser.execute_medium_delay()

                InputListener.listen()

                try:
                    p = WebDriverWait(driver, 20).until(
                        EC.presence_of_element_located(
                            (By.XPATH, ".//div[@class='content']//p[text()='唉，这份文件不在Sci-Hub数据库中。']"))
                    )

                    # 当前论文文件不存在于数据库，跳过
                    back_button = driver.find_element(By.XPATH, ".//div[@id='return']//a[text()='← 回到主页']")
                    back_button.click()
                    print(f"{i}: {search_title} 未能找到该论文pdf")
                    continue

                except TimeoutException:
                    pass

                DelayParser.execute_short_delay()

                pdf_url = driver.find_element(By.XPATH, ".//button[text()='↓ 下载']").get_property("onclick")
                pdf_url = f'{target_url}/{pdf_url.replace("location.href=", "")}'

                # pdf_url = driver.find_element(By.XPATH, ".//embed").get_property("src")
                # # 去掉URL中的锚点部分
                #
                # if "#" in pdf_url:
                #     pdf_url = pdf_url.split("#")[0]

                resp = requests.get(pdf_url)
                if resp.status_code == 200:
                    # 下载pdf文件
                    base_url = "D:/Users/Z/PycharmProjects/MyBlogFlask"
                    pdf_path = f"{base_url}{file_path.lstrip('.')}/{search_title}.pdf"
                    with open(pdf_path, "wb") as f:
                        f.write(resp.content)
                    print(f"{i}: {search_title} 论文pdf下载成功")
                else:
                    # 当前论文文件下载失败，返回主页，跳过
                    print(f"{i}: {search_title} 论文pdf下载失败")

                driver.back()

            print("爬虫程序正常结束")

        except Exception as e:
            print(f"[WebSpiderParser/crawl_papers_pdf_original] 爬虫抛出错误: {e}")

        finally:

            if pause_function:
                InputListener.deactivate()

            # 关闭浏览器
            driver.quit()

    # TODO: 问题1：少数网站进入了"我是人类"验证，无法看到摘要
    # TODO: 问题2：多次进入后sciencedirect提示出错，无法看到摘要
    # TODO: 问题3：user data无法复制
    # TODO 用endnote插件下载pdf
    # TODO 多线程开多个浏览器分别执行爬虫
    @classmethod
    def crawl_papers_pdf_original_by_scholar_engine(cls, target_url: str, search_box_xpath: str,
                                                    search_button_xpath: str, pause_function: bool = False,
                                                    file_path=None, last_file_path=None):

        if pause_function:
            InputListener.activate()

        driver = cls.configure(
            is_random_user_agent=False,
            is_headless=False
        )

        try:
            # 打开网站
            driver.get(target_url)
            DelayParser.execute_short_delay()

            InputListener.listen()

            # 读取数据源文件，转变为列表
            df = pd.read_csv(f"{last_file_path}.csv", index_col=False)
            paper_title_list = df.loc[:, "title"].tolist()

            for i, search_title in enumerate(paper_title_list):

                # 搜索框输入
                search_box = driver.find_element(By.XPATH, search_box_xpath)
                # 清空输入框内容
                search_box.clear()
                search_box.send_keys(search_title)
                DelayParser.execute_short_delay()

                # 点击搜索按钮
                search_button = driver.find_element(By.XPATH, search_button_xpath)
                search_button.click()
                DelayParser.execute_short_delay()

                InputListener.listen()

                DelayParser.execute_short_delay()

                # 进入第一个网页链接
                try:
                    a = WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.XPATH, ".//h3[@class='gs_rt']//a"))
                    )

                except TimeoutException:
                    # 无法找到当前论文所在网页，跳过
                    print("Not found!")
                    continue

                cls.direct_click_element(driver, a)
                DelayParser.execute_medium_delay()

                try:
                    cls.switch_to_latest_window(driver)

                    # 切换到iframe
                    iframe = WebDriverWait(driver, 20).until(
                        EC.presence_of_element_located((By.XPATH, ".//iframe[@srcdoc]"))
                    )
                    driver.switch_to.frame(iframe)

                    view_pdf = driver.find_element(By.XPATH, ".//div[text()='查看PDF']")

                    view_pdf.click()

                    cls.switch_to_latest_window(driver)
                    DelayParser.execute_short_delay()

                    download_link_url = ""
                    for _ in range(5):
                        download_link = WebDriverWait(driver, 30).until(
                            EC.presence_of_element_located((By.XPATH, ".//a[@id='download-link']"))
                        )
                        download_link_url = download_link.get_property("href")

                        DelayParser.execute_short_delay()

                        if download_link_url != "":
                            break

                except Exception as e:

                    driver.close()
                    window_len = cls.switch_to_latest_window(driver)
                    if window_len > 1:
                        driver.close()
                        cls.switch_to_latest_window(driver)
                    driver.back()
                    driver.get(target_url)

                    continue

                resp = requests.get(download_link_url)
                if resp.status_code == 200:
                    # 下载pdf文件
                    base_url = "D:/Users/Z/PycharmProjects/MyBlogFlask"
                    pdf_path = f"{base_url}{file_path.lstrip('.')}/{search_title}.pdf"
                    with open(pdf_path, "wb") as f:
                        f.write(resp.content)
                    print(f"{i}: {search_title} 论文pdf下载成功")
                else:
                    # 当前论文文件下载失败，返回主页，跳过
                    print(f"{i}: {search_title} 论文pdf下载失败")

                driver.close()
                window_len = cls.switch_to_latest_window(driver)
                if window_len > 1:
                    driver.close()
                    cls.switch_to_latest_window(driver)
                driver.back()
                driver.get(target_url)

            print("爬虫程序正常结束")

        except Exception as e:
            print(f"[WebSpiderParser/crawl_papers_pdf_original] 爬虫抛出错误: {e}")

        finally:

            if pause_function:
                InputListener.deactivate()

            # 关闭浏览器
            driver.quit()

    @classmethod
    def crawl_summary_from_llm_web(cls, target_url: str, search_box_xpath: str, search_button_xpath: str,
                                   pause_function: bool = False):

        if pause_function:
            InputListener.activate()

        driver = cls.configure(
            is_random_user_agent=False,
            is_headless=False
        )

        try:
            # 打开网站
            driver.get(target_url)
            DelayParser.execute_long_delay()

            # TODO:强制暂停程序

            InputListener.listen()

            # 读取数据源文件，转变为列表
            df = pd.read_csv("./abstract_papers.csv")

            paper_id_list = df.iloc[:, 0].tolist()
            paper_title_list = df.loc[:, "title"].tolist()

            paper_id_title_dict = dict(zip(paper_id_list, paper_title_list))

            paper_llm_summary_df = copy.deepcopy(df)
            for paper_id, paper_title in paper_id_title_dict.items():

                # 判断论文PDF文件是否存在
                if not os.path.exists(f"./storage/{paper_id}.pdf"):
                    continue

                # 复制文件到粘贴板
                pdf_file_path = f"D:/Users/Z/PycharmProjects/MyBlogFlask/storage/{paper_id}.pdf"
                # 直接调用 PowerShell
                command = [
                    "powershell",
                    "-sta",
                    f"$sc=New-Object System.Collections.Specialized.StringCollection;"
                    f"$sc.Add('{pdf_file_path}');"
                    "Add-Type -Assembly 'System.Windows.Forms';"
                    "[System.Windows.Forms.Clipboard]::SetFileDropList($sc);"
                ]
                subprocess.run(command, shell=True)
                DelayParser.execute_short_delay()

                # 开始新对话
                new_start_button = driver.find_element(By.XPATH, ".//div[@class='c7dddcde']")
                new_start_button.click()
                DelayParser.execute_short_delay()

                # 搜索框粘贴文件并输入
                search_box = driver.find_element(By.XPATH, search_box_xpath)
                search_box.click()
                search_box.send_keys(Keys.CONTROL, 'v')
                DelayParser.execute_long_delay()

                InputListener.listen()

                prompt = "Please give the research question, research methodology, contributions and limitations of the uploaded PDF paper in JSON format"

                search_box.send_keys(prompt)
                DelayParser.execute_delay(30)

                InputListener.listen()

                # 点击搜索按钮
                search_button = cls.wait_until_element(driver, search_button_xpath)
                search_button.click()
                DelayParser.execute_delay(30)

                InputListener.listen()

                try:
                    code_block = driver.find_element(By.XPATH, ".//div[@class='md-code-block']")
                    json_data = json.loads(code_block.text.replace("\n", "").replace("json复制", ""))
                except Exception:
                    print(f"{paper_id}: {paper_title} 获取summary失败，跳过")
                    continue

                # 修正可能出现的英文单复数问题
                if "research_questions" in json_data.keys():
                    paper_llm_summary_df.loc[paper_id, "research_question"] = str(json_data["research_questions"])
                if "research_methodologies" in json_data.keys():
                    paper_llm_summary_df.loc[paper_id, "research_methodology"] = str(
                        json_data["research_methodologies"])
                if "contribution" in json_data.keys():
                    paper_llm_summary_df.loc[paper_id, "contributions"] = str(json_data["contribution"])
                if "limitation" in json_data.keys():
                    paper_llm_summary_df.loc[paper_id, "limitations"] = str(json_data["limitation"])

                if "research_question" in json_data.keys():
                    paper_llm_summary_df.loc[paper_id, "research_question"] = str(json_data["research_question"])
                if "research_methodology" in json_data.keys():
                    paper_llm_summary_df.loc[paper_id, "research_methodology"] = str(json_data["research_methodology"])
                if "contributions" in json_data.keys():
                    paper_llm_summary_df.loc[paper_id, "contributions"] = str(json_data["contributions"])
                if "limitations" in json_data.keys():
                    paper_llm_summary_df.loc[paper_id, "limitations"] = str(json_data["limitations"])

                print(f"{paper_id}: {paper_title}: 获取summary成功")

                DelayParser.execute_short_delay()

            # 保存到CSV文件
            paper_llm_summary_df = paper_llm_summary_df.iloc[:, 1:]
            paper_llm_summary_df.to_csv("./summary_papers.csv", index=True, encoding="utf-8-sig")

            print("爬虫程序正常结束")

        except Exception as e:
            print(f"[WebSpiderParser/crawl_papers_pdf_original] 爬虫抛出错误: {e}")

        finally:

            if pause_function:
                InputListener.deactivate()

            # 关闭浏览器
            driver.quit()

    # TODO: 引用自动下载网站：http://git.macropus.org/citation-finder
