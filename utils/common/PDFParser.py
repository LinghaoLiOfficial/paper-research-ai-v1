from langchain_community.document_loaders import PyPDFLoader
import re

from utils.llm.APILLMParser import APILLMParser


class PDFParser:

    @classmethod
    def load(cls, pdf_path):
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()

        total_text = "\n".join([page.page_content for page in pages])
        total_text = total_text.replace("\n\n", "\n")

        return total_text

    @classmethod
    def split_and_classify(cls, text):
        # 正则表达式匹配前后都是 \n，并且长度小于等于 80 的子字符串
        pattern = re.compile(r"(?<=\n)(.{1,80}?)(?=\n)")
        matches = [match.group() for match in pattern.finditer(text) if len(match.group().strip()) <= 80]

        # 获取论文的子标题列表
        system_prompt = "You are asked to select from a list of raw strings provided by the user the strings that are the titles of the chapters of the complete thesis and store them in a list and then output them in JSON format."
        user_prompt = str(matches)

        max_retries = 5
        min_title_num = 4

        title_list = []
        for i in range(max_retries):
            output = APILLMParser.call_llm_for_get_json(
                model="deepseek",
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )

            title_list = output.values()
            title_list = [x for x in title_list][0]

            if len(title_list) >= min_title_num:
                break

        # 判断子标题是否与实验相关
        system_prompt1 = "Determine whether each subheading in the list of paper subheadings provided by the user describes the concrete experiments (not theory) in the paper in JSON format."
        user_prompt1 = str(title_list)

        judge_dict = APILLMParser.call_llm_for_get_json(
            model="deepseek",
            system_prompt=system_prompt1,
            user_prompt=user_prompt1
        )

        new_section_list = []
        is_init = True
        for k, v in judge_dict.items():
            if is_init:
                new_section_list.append(k)
                is_init = False
                continue
            if v:
                break
            new_section_list.append(k)

        parts = text.split("\n")

        # 构造字典列表
        results = []
        current_content = ""
        for part in parts:
            if part in title_list:
                if current_content != "":
                    results.append({
                        "type": "content",
                        "text": current_content
                    })
                    current_content = ""
                results.append({
                    "type": "title",
                    "text": part.strip()
                })
            else:
                current_content += part.strip()

        # 修正标题
        for i in range(len(results)):
            if results[i]["type"] == "title":
                break
            results[i]["type"] = "title"

        filtered_results = []
        i = 0
        for result in results:
            if i >= 2:
                if result["type"] == "title" and result["text"] not in new_section_list:
                    break

            if result["type"] == "content":
                filtered_results.append(result)

            i += 1

        total_text = " ".join([x['text'] for x in filtered_results])

        return total_text


