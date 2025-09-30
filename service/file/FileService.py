from flask import send_from_directory

from utils.common.FileSaver import FileSaver


class FileService:

    # 根据文件路径，获取文件
    @classmethod
    def show_file(cls, file_path: str):

        file_name = file_path.split("/")[-1]
        base_path = f'./{file_path.replace(f"/{file_name}", "")}'

        return send_from_directory(base_path, file_name, as_attachment=False)
