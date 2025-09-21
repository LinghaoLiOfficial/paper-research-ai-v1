from flask import jsonify, send_file

from entity.common.StrucResult import StrucResult


class Resp(StrucResult):

    @classmethod
    def to_download_file(cls, file_path: str):
        return send_file(
            file_path,
            as_attachment=True
        )

    @classmethod
    def to_show_file(cls, file_path: str, file_type: str):
        return send_file(
            file_path,
            as_attachment=False,
            # mimetype=file_type,
            download_name=file_path.split("/")[-1]
        )

    def to_json(self):
        return jsonify(self.to_dict())

    @classmethod
    def _build(cls, check: bool, code: int, message: str, data: dict):
        return cls(check, code, message, data).to_json()

    @classmethod
    def build_success(cls, code: int = None, message: str = "", data: dict = {}):
        return cls._build(
            check=True,
            code=code,
            message=message,
            data=data
        )

    @classmethod
    def build_error(cls, code: int = None, message: str = "", data: dict = {}):
        return cls._build(
            check=False,
            code=code,
            message=message,
            data=data
        )

    @classmethod
    def build_db_error(cls):
        return cls.build_error(
            code=60001,
            message="数据库操作执行失败"
        )

    @classmethod
    def build_jwt_error(cls, jwt_parser_result: StrucResult):
        return cls.build_error(
            code=jwt_parser_result.code,
            message=jwt_parser_result.message
        )

