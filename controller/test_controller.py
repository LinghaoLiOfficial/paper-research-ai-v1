import os

from flask import Blueprint
from flask import Flask, request, jsonify, send_from_directory, render_template
import uuid
import time
from PIL import Image

from service.TestService import TestService

from entity.common.Req import Req
from entity.common.Resp import Resp
from utils.common.JWTParser import JWTParser
from utils.common.RandomStrGenerator import RandomStrGenerator
from utils.common.TimeParser import TimeParser
from utils.database.SqlDriver import SqlDriver

# 实例化/test的Blueprint
test_bp = Blueprint(
    name="test",
    import_name=__name__,
    url_prefix="/test"
)


@test_bp.post("/uploadImages")
def upload_images():
    token = Req.receive_header_token()

    # 解析token
    jwt_parser_result = JWTParser.decode_user_id(token=token)
    if not jwt_parser_result.check:
        return Resp.build_jwt_error(jwt_parser_result)

    user_id = jwt_parser_result.get_data_on_results()
    files = request.files.getlist('images[]')
    FILE_PATH = "./storage/{}/knowledge_base/{}"

    # 支持的图像格式
    SUPPORTED_IMAGE_TYPES = ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp']

    for file in files:
        try:
            file_id = RandomStrGenerator.generate_uuid()
            timestamp = TimeParser.get_current_time()

            # 获取安全的文件名
            filename = file.filename
            file_ext = filename.split('.')[-1].lower() if '.' in filename else ''

            # 创建存储目录
            user_dir = os.path.dirname(FILE_PATH.format(user_id, ""))
            os.makedirs(user_dir, exist_ok=True)

            # 数据库插入
            sql = """INSERT INTO my_blog_mysql.file 
                     (file_id, file_name, file_size, file_type, owner_id, create_timestamp, belong_folder, file_label) 
                     VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"""

            base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename

            SqlDriver.execute_write(sql, {
                "file_id": file_id,
                "file_name": base_name,
                "file_size": 0,  # 临时设置为0，稍后更新
                "file_type": file_ext,
                "owner_id": user_id,
                "create_timestamp": timestamp,
                "belong_folder": "default",
                "file_label": "im"
            })

            # 保存原始文件
            orig_path = f"{FILE_PATH.format(user_id, file_id)}.{file_ext}"
            file.save(orig_path)
            orig_size = os.path.getsize(orig_path)

            # === 生成预览图 ===
            if file_ext in SUPPORTED_IMAGE_TYPES:
                try:
                    # 创建预览图路径
                    preview_path = f"{FILE_PATH.format(user_id, file_id)}_preview.jpg"

                    # 设置预览图参数
                    PREVIEW_WIDTH = 800  # 预览图最大宽度
                    JPEG_QUALITY = 40  # 压缩质量 (0-100)

                    # 使用Pillow处理图像
                    img = Image.open(orig_path)

                    # 处理透明通道（将RGBA转换为RGB）
                    if img.mode in ('RGBA', 'LA'):
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[-1])
                        img = background

                    # 保持宽高比调整尺寸
                    w_percent = PREVIEW_WIDTH / float(img.size[0])
                    h_size = int(float(img.size[1]) * float(w_percent))
                    img = img.resize((PREVIEW_WIDTH, h_size), Image.LANCZOS)

                    # 转换为RGB模式（确保兼容JPEG）
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    # 保存预览图
                    img.save(
                        preview_path,
                        "JPEG",
                        quality=JPEG_QUALITY,
                        optimize=True,
                        progressive=True  # 渐进式加载
                    )

                    # 可选：在数据库中记录预览图信息
                    # (如果需要在数据库中单独记录预览图，可以在此添加额外SQL操作)

                except Exception as img_e:
                    # 记录错误但继续处理其他文件
                    print(f"生成预览图失败: {str(img_e)}")

        except Exception as e:
            print(f"文件处理失败: {str(e)}")
            # 继续处理其他文件
            continue

    return Resp.build_success()


@test_bp.get("/getAllImages")
def get_all_images():
    sql = "SELECT * from my_blog_mysql.file WHERE file_label = %s ORDER BY create_timestamp DESC"
    results = SqlDriver.execute_read(sql, {
        "file_label": "im"
    })

    result_list = results.get_data_on_results()

    FILE_PATH = "./storage/{}/knowledge_base/{}"

    images = []
    for result in result_list:
        sql1 = "SELECT * FROM my_blog_mysql.user WHERE user_id = %s"
        user_results = SqlDriver.execute_read(sql1, {
            "user_id": result["owner_id"]
        })

        images.append({
            'id': result["file_id"],  # 使用UUID作为ID
            'name': result["file_name"],
            'thumbnail': f'/file/{FILE_PATH.format(result["owner_id"], result["file_id"]).lstrip("./")}.{result["file_type"]}',
            'large': f'/file/{FILE_PATH.format(result["owner_id"], result["file_id"]).lstrip("./")}_preview.jpg',
            'date': TimeParser.convert_time_format(result["create_timestamp"]),
            'user': [_ for _ in user_results.get_data_on_results()][0]["username"] or ""
        })

    return Resp.build_success(data={
        "images": images
    })


@test_bp.post("/deleteImages")
def delete_images():
    file_id_list = Req.receive_post_param("fileIdList")

    FILE_PATH = "./storage/{}/knowledge_base/{}"
    for file_id in file_id_list:

        sql2 = "SELECT * FROM my_blog_mysql.file WHERE file_id = %s"
        results = SqlDriver.execute_read(sql2, {
            "file_id": file_id
        })

        user_id = [_ for _ in results.get_data_on_results()][0]["owner_id"]
        file_type = [_ for _ in results.get_data_on_results()][0]["file_type"]

        file_path = f"{FILE_PATH.format(user_id, file_id)}.{file_type}"
        try:
            os.remove(file_path)

            sql3 = "DELETE FROM my_blog_mysql.file WHERE file_id = %s"
            SqlDriver.execute_write(sql3, {
                "file_id": file_id
            })
        except Exception as e:
            print(e)

    return Resp.build_success()


# 【GET】【/ping】测试是否连通
@test_bp.get("/ping")
def ping_api():
    return Resp.build_success(
        message="Ping Success!"
    )


# 【GET】【/connectMySQL】测试是否连通MySQL数据库
@test_bp.get("/connectMySQL")
def connect_mysql_api():
    return TestService.connect_mysql()


# 【GET】【/connectNeo4j】测试是否连通Neo4j数据库
@test_bp.get("/connectNeo4j")
def connect_neo4j_api():
    return TestService.connect_neo4j()
