from flask import Blueprint

from entity.common.Req import Req
from service.KnowledgeBaseService import KnowledgeBaseService

# 实例化Blueprint
kb_bp = Blueprint(
    name="kb",
    import_name=__name__,
    url_prefix="/kb"
)


# 创建文件
@kb_bp.post("/docCreateFile")
def doc_create_file_api():
    file_type = Req.receive_post_param("fileType")
    file_name = Req.receive_post_param("fileName")
    token = Req.receive_header_token()

    return KnowledgeBaseService.doc_create_file(
        file_type=file_type,
        file_name=file_name,
        token=token
    )


# 获取上传所在的文件夹列表
@kb_bp.get("/getUploadFolderOptions")
def get_upload_folder_options_api():
    token = Req.receive_header_token()

    return KnowledgeBaseService.get_upload_folder_options(
        token=token
    )


# 获取我的空间文件列表
@kb_bp.get("/docGetMySpaceFiles")
def doc_get_my_space_files_api():
    token = Req.receive_header_token()
    belong_folder = Req.receive_get_param("belongFolder")

    return KnowledgeBaseService.doc_get_my_space_files(
        belong_folder=belong_folder,
        token=token
    )


# 获取知识图谱
@kb_bp.get("/getKnowledgeGraph")
def get_knowledge_graph_api():
    file_id = Req.receive_get_param("fileId")
    token = Req.receive_header_token()

    return KnowledgeBaseService.get_knowledge_graph(
        file_id=file_id,
        token=token
    )


# 更新知识图谱
@kb_bp.post("/updateKnowledgeGraph")
def update_knowledge_graph_api():
    graph_data = Req.receive_post_param("graphData")
    token = Req.receive_header_token()

    return KnowledgeBaseService.update_knowledge_graph(
        graph_data=graph_data,
        token=token
    )


# 更新知识图谱图片
@kb_bp.post("/uploadKnowledgeGraphImage")
def upload_knowledge_graph_image_api():
    image = Req.receive_file_param("file")
    file_id = Req.receive_form_param("id")
    token = Req.receive_header_token()

    return KnowledgeBaseService.upload_knowledge_graph_image(
        image=image,
        file_id=file_id,
        token=token
    )


# 加载PDF文件
@kb_bp.get("/docLoadPDFFile")
def doc_load_pdf_file_api():
    file_id = Req.receive_get_param("fileId")
    token = Req.receive_header_token()

    return KnowledgeBaseService.doc_load_pdf_file(
        file_id=file_id,
        token=token
    )


# 翻译文本
@kb_bp.get("/docTranslateText")
def doc_translate_text_api():
    str_to_translate = Req.receive_get_param("strToTranslate")
    translate_source = Req.receive_get_param("translateSource")

    return KnowledgeBaseService.doc_translate_text(
        str_to_translate=str_to_translate,
        translate_source=translate_source
    )


# 加载md文件
@kb_bp.get("/editLoadMarkdownFile")
def edit_load_markdown_file_api():
    file_id = Req.receive_get_param("fileId")
    token = Req.receive_header_token()

    return KnowledgeBaseService.edit_load_markdown_file(
        file_id=file_id,
        token=token
    )


# 保存md文件
@kb_bp.post("/editSaveMarkdownFile")
def edit_save_markdown_file_api():
    content = Req.receive_post_param("content")
    file_id = Req.receive_post_param("fileId")
    token = Req.receive_header_token()

    return KnowledgeBaseService.edit_save_markdown_file(
        content=content,
        file_id=file_id,
        token=token
    )


# 更新md文件中的图片
@kb_bp.post("/uploadMarkdownImage")
def upload_markdown_image_api():
    image = Req.receive_file_param("image")
    file_id = Req.receive_form_param("fileId")
    token = Req.receive_header_token()

    return KnowledgeBaseService.upload_markdown_image(
        image=image,
        file_id=file_id,
        token=token
    )


# 更新PDF文件
@kb_bp.post("/uploadPDFFile")
def upload_pdf_file_api():
    pdf_list = Req.receive_files_param()
    upload_folder = Req.receive_form_param("uploadFolder")
    token = Req.receive_header_token()

    return KnowledgeBaseService.upload_pdf_file(
        pdf_list=pdf_list,
        upload_folder=upload_folder,
        token=token
    )


# 删除文件
@kb_bp.post("/docDeleteFile")
def doc_delete_file_api():
    file_type = Req.receive_post_param("fileType")
    file_id = Req.receive_post_param("fileId")
    token = Req.receive_header_token()

    return KnowledgeBaseService.doc_delete_file(
        file_type=file_type,
        file_id=file_id,
        token=token
    )


# 获取总文件大小
@kb_bp.get("/getTotalFileSize")
def get_total_file_size_api():
    token = Req.receive_header_token()

    return KnowledgeBaseService.get_total_file_size(
        token=token
    )


# 更新PDF文件
@kb_bp.post("/docUpdatePDFFile")
def doc_update_pdf_file_api():
    pdf = Req.receive_file_param("file")
    file_size = Req.receive_form_param("fileSize")
    file_id = Req.receive_form_param("fileId")
    token = Req.receive_header_token()

    return KnowledgeBaseService.doc_update_pdf_file(
        pdf=pdf,
        file_size=file_size,
        file_id=file_id,
        token=token
    )
