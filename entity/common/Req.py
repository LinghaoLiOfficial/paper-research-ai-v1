from flask import request


class Req:
    @classmethod
    def receive_get_param(cls, param: str):
        return request.args.get(param)

    @classmethod
    def receive_post_param(cls, param: str):
        return request.get_json().get(param)

    @classmethod
    def receive_header_token(cls):
        return request.headers.get('Authorization')

    @classmethod
    def receive_file_param(cls, param: str):
        return request.files.get(param)

    @classmethod
    def receive_files_param(cls):
        return request.files.getlist('files[]')

    @classmethod
    def receive_form_param(cls, param: str):
        return request.form.get(param)

