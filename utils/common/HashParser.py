import bcrypt
import hashlib


class HashParser:
    @classmethod
    def generate_salt(cls):
        return bcrypt.gensalt().decode()

    @classmethod
    def hash_encode(cls, original_string):
        # 创建一个 SHA-256 哈希对象
        hash_object = hashlib.sha256()

        # 更新哈希对象，传入需要加密的字符串（需要编码为字节）
        hash_object.update(original_string.encode('utf-8'))

        # 获取加密后的哈希值（十六进制表示）
        hashed_string = hash_object.hexdigest()

        return hashed_string
