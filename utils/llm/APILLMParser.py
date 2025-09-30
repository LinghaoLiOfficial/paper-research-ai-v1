import tiktoken
from openai import OpenAI
from zhipuai import ZhipuAI
import os
import re
import json


class APILLMParser:
    STATUS = False

    GPT_MODEL = None
    DEEPSEEK_MODEL = None
    ZHIPU_MODEL = None
    ZHIPU_MODEL_AGENT = None

    TOKEN_CALCULATE_MODEL = None

    @classmethod
    def init(cls):
        if not cls.STATUS:
            cls.GPT_MODEL = OpenAI(api_key=os.getenv("VIP_GPT_API_KEY"), base_url=os.getenv("OPENAI_API_URL"))
            cls.DEEPSEEK_MODEL = OpenAI(api_key=os.getenv("DEEPSEEK_V3_API_KEY"), base_url=os.getenv("DEEPSEEK_API_URL"))
            cls.ZHIPU_MODEL = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))
            cls.ZHIPU_MODEL_AGENT = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"), base_url=os.getenv("ZHIPU_API_URL"))

            cls.TOKEN_CALCULATE_MODEL = tiktoken.encoding_for_model("gpt-4")

            cls.STATUS = True

    @classmethod
    def calculate_token(cls, string):
        cls.init()

        return len(cls.TOKEN_CALCULATE_MODEL.encode(string))

    # Zhipu
    @classmethod
    def search_paper_call_zhipu_llm_api(cls, keywords):
        cls.init()

        completion = cls.ZHIPU_MODEL_AGENT.assistant.conversation(
            assistant_id="659e54b1b8006379b4b2abd6",
            conversation_id=None,
            model="glm-4-assistant",
            messages=[
                {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": f"在中国知网检索以下关键词以获取100篇相关性高的中文论文的doi: [{keywords}]"
                    }]
                }
            ],
            stream=True,
            attachments=None,
            metadata=None
        )

        # TODO

        list = [x for x in completion]

        for x in list:
            print(x)

        output_text = completion.choices[0].message.content

        try:
            output_dict = eval(output_text)
        except Exception as e:
            output_dict = ""

        return output_dict

    # Zhipu
    @classmethod
    def call_assistant_llm_api(cls, text):
        cls.init()

        completion = cls.ZHIPU_MODEL_AGENT.assistant.conversation(
            assistant_id="65a265419d72d299a9230616",
            conversation_id=None,
            model="glm-4-assistant",
            messages=[
                {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": text
                    }]
                }
            ],
            stream=True,
            attachments=None,
            metadata=None
        )

        completion_list = [x for x in completion]

        return completion_list

    @classmethod
    def call_embedding_llm_api(cls, str_list: list):
        cls.init()

        completion = cls.ZHIPU_MODEL.embeddings.create(
            model="embedding-3",
            input=str_list
        )
        output_list = [x.embedding for x in completion.data]

        return output_list

    @classmethod
    def call_llm_api(cls, system_prompt="", user_prompt="", messages=[], resp_format="text", model_type="zhipu", model="glm-4-long", will_clear_invalid_char=False):

        cls.init()

        # 文档读取后的文本可能包含无效的字符
        if will_clear_invalid_char:
            def remove_surrogates(text):
                return ''.join(
                    char for char in text
                    if not ('\ud800' <= char <= '\udfff')
                )

            def safe_json_dumps(data):
                # 清理数据中的代理对
                cleaned_data = {
                    k: remove_surrogates(v) if isinstance(v, str) else v
                    for k, v in data.items()
                }
                return cleaned_data

            # 清理无效数据
            messages = [safe_json_dumps(message) for message in messages]

        if messages == []:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

        if model_type == "zhipu":
            """
                model:
                    glm-4-plus
                    glm-4-long
            """

            if resp_format == "text":
                completion = cls.ZHIPU_MODEL.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=False
                )
            else:
                completion = cls.ZHIPU_MODEL.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=False,
                    response_format={
                        'type': 'json_object'
                    }
                )

                output_text = completion.choices[0].message.content

                try:
                    output_dict = eval(output_text)
                except Exception as e:
                    print(e)
                    output_dict = None

                return output_dict

        elif model_type == "deepseek":
            """
                model:
                    deepseek-chat
                    deepseek-reasoner
            """

            if resp_format == "text":
                completion = cls.DEEPSEEK_MODEL.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=False
                )
            else:

                if model == "deepseek-reasoner":
                    completion = cls.DEEPSEEK_MODEL.chat.completions.create(
                        model=model,
                        messages=messages,
                        stream=False,
                    )

                    output_text = completion.choices[0].message.content

                    output_text = output_text.replace("true", "True").replace("false", "False")

                    match = re.search(r'```json(.*?)```', output_text, re.DOTALL)
                    output_str = ""
                    if match:
                        output_str = match.group(1).strip()

                    try:
                        output_dict = eval(output_str)
                    except Exception as e:
                        print(e)
                        output_dict = None

                    return output_dict

                else:
                    completion = cls.DEEPSEEK_MODEL.chat.completions.create(
                        model=model,
                        messages=messages,
                        stream=False,
                        response_format={
                            'type': 'json_object'
                        }
                    )

                    output_text = completion.choices[0].message.content

                    try:
                        output_dict = eval(output_text)
                    except Exception as e:
                        print(e)
                        output_dict = None

                    return output_dict

        elif model_type == "gpt":
            """
                model:
                    gpt-3.5-turbo
            """

            if resp_format == "text":
                completion = cls.GPT_MODEL.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=False
                )
            else:
                completion = cls.GPT_MODEL.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=False
                )
                output_text = completion.choices[0].message.content

                try:
                    output_dict = eval(output_text)
                except Exception as e:
                    print(e)
                    output_dict = None

                return output_dict

        output_text = completion.choices[0].message.content

        if resp_format == "json":
            try:
                output_dict = eval(output_text)
            except Exception as e:
                print(e)
                output_dict = None

            return output_dict

        # resp_format 为 "text"
        return output_text

    @classmethod
    def call_llm_for_get_json(cls, model, system_prompt, user_prompt):

        max_retries = 5

        for i in range(max_retries):
            output = {
                "deepseek": lambda: cls.call_llm_api(system_prompt, user_prompt, resp_format="json", model_type="deepseek", model="deepseek-chat")
            }.get(model)()

            output = output.replace("true", "True").replace("false", "False")

            match = re.search(r'```json(.*?)```', output, re.DOTALL)
            if match:
                output_str = match.group(1).strip()
                try:
                    output_dict = eval(output_str)
                    return output_dict
                except Exception as e:
                    return None

            else:
                try:
                    output_dict = eval(output)
                    return output_dict
                except Exception as e:
                    return None

        return None
