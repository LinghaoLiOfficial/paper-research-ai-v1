import random
import string
import uuid
import os


class RandomStrGenerator:

    @classmethod
    def generate_5_random_str(cls):
        # 定义包含数字和大写字母的字符集
        characters = string.digits + string.ascii_uppercase
        # 随机选择字符并拼接成字符串
        random_string = ''.join(random.choice(characters) for _ in range(5))
        return random_string

    @classmethod
    def generate_uuid(cls):
        return str(uuid.uuid4())

    @classmethod
    def generate_validation_code(cls):
        return "".join([str(random.randint(0, 9)) for _ in range(int(os.getenv("VALIDATION_CODE_LEN")))])

