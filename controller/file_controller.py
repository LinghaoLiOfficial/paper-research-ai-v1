from flask import Blueprint

from service.file.FileService import FileService


# 实例化Blueprint
file_bp = Blueprint(
    name="file",
    import_name=__name__,
    url_prefix="/file"
)


# 获取任意本地文件
@file_bp.get("/<path:file_path>")
def get_file_api(file_path):

    return FileService.show_file(
        file_path=file_path
    )
