import deepl
import os
import json
import re

from entity.common.StrucResult import StrucResult
from utils.llm.APILLMParser import APILLMParser


class TextTranslator:

    @classmethod
    def translate(cls, target_str, translate_source="deepseek"):

        if target_str is None:
            text = ""
        else:

            try:

                text = ""
                if translate_source == "deepl":

                    auth_key = os.getenv("DEEPL_API_KEY")

                    translator = deepl.Translator(auth_key)

                    result = translator.translate_text(target_str, target_lang="ZH")

                    text = result.text

                elif translate_source == "zhipu":

                    system_prompt = "Please translate the English text provided by the user into Chinese"

                    text = APILLMParser.call_llm_api(
                        system_prompt=system_prompt,
                        user_prompt=target_str
                    )

                elif translate_source == "deepseek":

                    system_prompt = "将用户给出的英文翻译为中文，输出保持与输入相同的格式，输出仅含译文，不要含任何注释说明内容"

                    text = APILLMParser.call_llm_api(
                        system_prompt=system_prompt,
                        user_prompt=target_str,
                        model_type="deepseek",
                        model="deepseek-chat"
                    )

                # 使用正则表达式去除括号及括号内的内容
                text = re.sub(r'（注：[^）]*）', '', text)

            except Exception as e:
                print(e)
                return StrucResult.build_error()

        return StrucResult.build_success_with_results(text)
