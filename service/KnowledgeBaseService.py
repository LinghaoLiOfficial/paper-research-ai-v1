import os
import shutil

from config.name.knowledge_base.KnowledgeFileHeaderEnName import KnowledgeFileHeaderEnName
from config.name.knowledge_base.KnowledgeFileHeaderZhName import KnowledgeFileHeaderZhName
from entity.common.Resp import Resp
from mapper.KnowledgeBaseMapper import KnowledgeBaseMapper
from utils.common.GraphParser import GraphParser
from utils.common.JWTParser import JWTParser
from utils.common.RandomStrGenerator import RandomStrGenerator
from datetime import datetime

from utils.common.TextTranslator import TextTranslator
from utils.common.TimeParser import TimeParser


class KnowledgeBaseService:

    FILE_PATH = "./storage/{}/knowledge_base/{}"

    @classmethod
    def get_upload_folder_options(cls, token):

        # 解析token
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        mysql_result = KnowledgeBaseMapper.select_file_where_owner_id_and_file_type({
            "owner_id": user_id,
            "file_type": "folder"
        })

        folder_files = mysql_result.get_data_on_results()

        upload_folder_options = [{
            "id": i + 1,
            "label": folder['file_name'],
            "name": folder['file_id']
        } for i, folder in enumerate(folder_files)]

        # 增加默认文件夹
        upload_folder_options = [{
            "id": 0,
            "label": "桌面",
            "name": "default"
        }] + upload_folder_options

        return Resp.build_success(data={
            "uploadFolderOptions": upload_folder_options
        })

    @classmethod
    def doc_update_pdf_file(cls, pdf, file_size, file_id, token):

        # 解析token
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        # 保存pdf文件
        pdf_path = f"{cls.FILE_PATH.format(user_id, file_id)}.pdf"
        pdf.save(pdf_path)

        return Resp.build_success()

    @classmethod
    def get_total_file_size(cls, token):

        # 解析token
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        # 获取用户的权限索引
        mysql_result = KnowledgeBaseMapper.select_user({
            "user_id": user_id
        })
        if not mysql_result.check:
            return Resp.build_db_error()

        auth_index = mysql_result.get_data_on_results()[0]['user_auth']

        # 获取用户权限的文件空间
        mysql_result_1 = KnowledgeBaseMapper.select_auth_where_index({
            "auth_index": auth_index
        })
        if not mysql_result_1.check:
            return Resp.build_db_error()

        auth_space_capacity = mysql_result_1.get_data_on_results()[0]['auth_space_capacity']

        # 获取目前用户所有文件的总大小
        mysql_result_2 = KnowledgeBaseMapper.select_file_where_owner_id({
            "owner_id": user_id
        })
        if not mysql_result_2.check:
            return Resp.build_db_error()

        total_file_size = sum([x['file_size'] for x in mysql_result_2.get_data_on_results()])

        return Resp.build_success(data={
            "authSpaceCapacity": auth_space_capacity,
            "totalFileSize": total_file_size
        })

    @classmethod
    def doc_delete_file(cls, file_type, file_id, token):

        # 解析token
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        mysq_result = KnowledgeBaseMapper.select_file_where_file_id({
            "file_id": file_id
        })

        upload_folder = mysq_result.get_data_on_results()[0]['belong_folder']

        # 删除mysql数据库的文件
        mysql_result1 = KnowledgeBaseMapper.delete_file({
            "file_id": file_id
        })
        if not mysql_result1.check:
            return Resp.build_db_error()

        base_path = cls.FILE_PATH.format(user_id, "").rstrip("/{}")

        if upload_folder != "default":
            base_path = f"{base_path}/{upload_folder}"

        # 遍历指定路径下的所有文件和文件夹
        for sub_path in os.listdir(base_path):
            if file_id in sub_path:
                suffix = sub_path.replace(file_id, "")
                if suffix == "":
                    shutil.rmtree(f"{base_path}/{sub_path}")
                else:
                    os.remove(f"{base_path}/{sub_path}")

        # 图谱文件需要额外在neo4j数据库中删除节点树
        if file_type == "graph":
            KnowledgeBaseMapper.delete_all_file_node_with_node_children({
                "file_id": file_id
            })

        return Resp.build_success()

    @classmethod
    def upload_pdf_file(cls, pdf_list, upload_folder, token):

        # 解析token
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        for pdf in pdf_list:

            # 获取新的uuid
            file_id = RandomStrGenerator.generate_uuid()
            # 获取当前时间
            create_timestamp = TimeParser.get_current_time()

            file_type = "pdf"
            file_name = pdf.filename.rstrip('.pdf')

            file_size = 0

            # 在Mysql数据库中插入文件数据
            mysql_result = KnowledgeBaseMapper.insert_file({
                "file_id": file_id,
                "file_name": file_name,
                "file_size": file_size,
                "file_type": file_type,
                "owner_id": user_id,
                "create_timestamp": create_timestamp,
                "belong_folder": upload_folder
            })
            if not mysql_result.check:
                return Resp.build_db_error()

            # 保存pdf文件
            if upload_folder == "default":
                pdf_path = f"{cls.FILE_PATH.format(user_id, file_id)}.pdf"
            else:
                pdf_path = f"{cls.FILE_PATH.format(user_id, upload_folder)}/{file_id}.pdf"
            pdf.save(pdf_path)

        return Resp.build_success()

    @classmethod
    def upload_markdown_image(cls, image, file_id, token):

        # 解析token
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        # 将图片改为uuid
        image_name = RandomStrGenerator.generate_uuid()

        base_path = cls.FILE_PATH.format(user_id, file_id)
        # 创建存储md图片的文件夹
        if not os.path.exists(base_path):
            os.mkdir(base_path)

        # 保存图片文件
        image_path = f"{base_path}/{image_name}.{image.filename.split('.')[-1]}"
        image.save(image_path)

        image_url = f"/file/{image_path.lstrip('./')}"

        return Resp.build_success(data={
            "imageUrl": image_url
        })

    @classmethod
    def edit_load_markdown_file(cls, file_id, token):

        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        with open(f"{cls.FILE_PATH.format(user_id, file_id)}.md", "r", encoding="utf-8") as file:
            content = file.read()

        # 删除弃用的图片(最多3个图片)
        if os.path.exists(cls.FILE_PATH.format(user_id, file_id)):
            counter = 0
            if content != "":
                for image_name in os.listdir(cls.FILE_PATH.format(user_id, file_id)):
                    if counter >= 3:
                        break

                    if image_name not in content:
                        try:
                            os.remove(f"{cls.FILE_PATH.format(user_id, file_id)}/{image_name}")
                            counter += 1
                        except Exception:
                            pass

        return Resp.build_success(data={
            "content": content
        })

    @classmethod
    def edit_save_markdown_file(cls, content, file_id, token):

        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        # 保存md文件
        with open(f"{cls.FILE_PATH.format(user_id, file_id)}.md", "w", encoding="utf-8") as file:
            file.write(content)

        return Resp.build_success()

    @classmethod
    def doc_translate_text(cls, str_to_translate, translate_source):

        translate_source = "zhipu"

        text_translator_result = TextTranslator.translate(
            target_str=str_to_translate,
            translate_source=translate_source
        )
        if not text_translator_result.check:
            return Resp.build_error()

        translated_str = text_translator_result.get_data_on_results()

        return Resp.build_success(
            data={
                "translatedStr": translated_str
            }
        )

    @classmethod
    def doc_load_pdf_file(cls, file_id, token):

        # 解析token
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        pdf_path = f"{cls.FILE_PATH.format(user_id, file_id)}.pdf"

        pdf_url = f"/file/{pdf_path.lstrip('./')}"

        return Resp.build_success(data={
            "pdfUrl": pdf_url
        })

    @classmethod
    def upload_knowledge_graph_image(cls, image, file_id, token):

        # 解析token
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        # 获取思维导图路径
        base_path = cls.FILE_PATH.format(user_id, file_id)
        # 保存图片文件
        image_path = f"{base_path}/{image.filename}"
        image.save(image_path)

        return Resp.build_success()

    @classmethod
    def update_knowledge_graph(cls, graph_data, token):

        # 解析token
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        # 删除原有思维导图结构
        neo4j_result = KnowledgeBaseMapper.delete_all_file_node_children({
            "file_id": graph_data["id"]
        })
        if not neo4j_result.check:
            return Resp.build_db_error()

        # 根据递归来创建或更新思维导图的树结构
        GraphParser.recursive_create_or_update_tree(parent_node=graph_data)

        # # 删除弃用的图片(最多3个图片)
        # neo4j_result_1 = KnowledgeBaseMapper.match_all_image_node_on_file_id({
        #     "file_id": graph_data["id"]
        # })
        # if not neo4j_result_1.check:
        #     return Resp.build_db_error()
        #
        # reserve_image_list = [x['image_name'] for x in neo4j_result_1.get_data_on_results()]
        # counter = 0
        # for image_name in os.listdir(cls.FILE_PATH.format(user_id, graph_data["id"])):
        #     if counter >= 3:
        #         break
        #
        #     if image_name not in reserve_image_list:
        #         try:
        #             os.remove(f"{cls.FILE_PATH.format(user_id, graph_data['id'])}/{image_name}")
        #             counter += 1
        #         except Exception:
        #             pass

        return Resp.build_success()

    @classmethod
    def get_knowledge_graph(cls, file_id, token):

        # 解析token
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        # 获取所有文本节点
        neo4j_result = KnowledgeBaseMapper.match_all_text_node_on_file_id({
            "file_id": file_id
        })
        if not neo4j_result.check:
            return Resp.build_db_error()

        # 获取所有文件节点
        neo4j_result_1 = KnowledgeBaseMapper.match_file_node_on_file_id({
            "file_id": file_id
        })
        if not neo4j_result_1.check:
            return Resp.build_db_error()

        # 获取所有图片节点
        neo4j_result_2 = KnowledgeBaseMapper.match_all_image_node_on_file_id({
            "file_id": file_id
        })
        if not neo4j_result_2.check:
            return Resp.build_db_error()

        # 递归获取树结构
        graph_data = GraphParser.init_recursive_get_tree(
            root_list=neo4j_result_1.get_data_on_results(),
            text_list=neo4j_result.get_data_on_results(),
            image_list=neo4j_result_2.get_data_on_results(),
            image_base_path=cls.FILE_PATH.format(user_id, file_id)
        )

        return Resp.build_success(data={
            "graphData": graph_data
        })

    @classmethod
    def doc_get_my_space_files(cls, belong_folder, token):

        # 解析token
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        # 默认为桌面
        if belong_folder is None:
            belong_folder = "default"

        # 根据user_id获取所有文件
        mysql_result = KnowledgeBaseMapper.select_file_where_user_id_order_by_time_desc({
            "user_id": user_id,
            "belong_folder": belong_folder
        })
        if not mysql_result.check:
            return Resp.build_db_error()

        # 获取表头的数据库属性名列表
        my_space_files_headers_property_name = KnowledgeFileHeaderEnName.to_list()

        # 获取表头名列表
        my_space_files_headers = [{
            "id": i,
            "label": label,
            "name": my_space_files_headers_property_name[i]
        } for i, label in enumerate(KnowledgeFileHeaderZhName.to_list())]

        # 获取表格行数据
        my_space_files = [
            {
                k: TimeParser.convert_time_format(v) if k == KnowledgeFileHeaderEnName.CREATE_TIMESTAMP else v for k, v
                in file.items()
            }
            for file in mysql_result.get_data_on_results()
        ]

        my_space_files = sorted(my_space_files, key=lambda v: v['file_label'], reverse=True)

        my_space_files = [{
            "file_id": "default",
            "file_name": "...",
            "file_type": "folder",
            "owner_id": user_id
        }] + my_space_files

        return Resp.build_success(data={
            "mySpaceFiles": my_space_files,
            "mySpaceFilesHeaders": my_space_files_headers
        })

    @classmethod
    def doc_create_file(cls, file_type, file_name, token):

        # 解析token
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        # 获取新的uuid
        file_id = RandomStrGenerator.generate_uuid()
        # 获取当前时间
        create_timestamp = TimeParser.get_current_time()

        file_size = 0

        # 在Mysql数据库中插入文件数据
        mysql_result = KnowledgeBaseMapper.insert_file({
            "file_id": file_id,
            "file_name": file_name,
            "file_size": file_size,
            "file_type": file_type,
            "owner_id": user_id,
            "create_timestamp": create_timestamp,
            "belong_folder": "default"
        })
        if not mysql_result.check:
            return Resp.build_db_error()

        if file_type == 'graph':
            # 获取文件夹路径
            base_path = cls.FILE_PATH.format(user_id, file_id)
            # 如果不存在文件夹则创建
            if not os.path.exists(base_path):
                os.makedirs(base_path)

            # 在neo4j数据库中创建新的文件节点
            neo4j_result = KnowledgeBaseMapper.merge_file_node({
                "file_type": file_type,
                "file_name": file_name,
                "user_id": user_id,
                "file_id": file_id
            })
            if not neo4j_result.check:
                return Resp.build_db_error()

        elif file_type == 'md':
            # 获取文件路径
            base_path = f"{cls.FILE_PATH.format(user_id, file_id)}.md"
            # 如果不存在文件则创建
            if not os.path.exists(base_path):
                with open(base_path, "w", encoding="utf-8") as file:
                    file.write("")

        elif file_type == 'folder':
            # 获取文件夹路径
            base_path = cls.FILE_PATH.format(user_id, file_id)
            # 如果不存在文件夹则创建
            if not os.path.exists(base_path):
                os.makedirs(base_path)

        return Resp.build_success()
