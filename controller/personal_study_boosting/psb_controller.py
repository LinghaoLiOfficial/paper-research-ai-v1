from flask import Blueprint

from service.personal_study_boosting.PSBService import PSBService
from entity.common.Req import Req

psb_bp = Blueprint(
    name="personal_study_boosting",
    import_name=__name__,
    url_prefix="/psb"
)


@psb_bp.post("/deleteNode")
def delete_node_api():
    token = Req.receive_header_token()

    node_id = Req.receive_post_param("nodeId")

    return PSBService.delete_node(
        token=token,
        node_id=node_id
    )


@psb_bp.post("/updateNodeProperty")
def update_node_property_api():
    token = Req.receive_header_token()

    node_id = Req.receive_post_param("nodeId")
    node_key = Req.receive_post_param("nodeKey")
    node_value = Req.receive_post_param("nodeValue")

    return PSBService.update_node_property(
        token=token,
        node_id=node_id,
        node_key=node_key,
        node_value=node_value
    )


@psb_bp.get("/searchNode")
def search_node_api():
    token = Req.receive_header_token()

    text_to_search = Req.receive_get_param("textToSearch")

    return PSBService.search_node(
        token=token,
        text_to_search=text_to_search
    )


@psb_bp.get("/translate")
def translate_api():
    token = Req.receive_header_token()

    str_to_translate = Req.receive_get_param("strToTranslate")

    return PSBService.translate(
        token=token,
        str_to_translate=str_to_translate
    )


@psb_bp.get("/showPDF")
def show_pdf_api():
    token = Req.receive_header_token()

    file_id = Req.receive_get_param("fileId")

    return PSBService.show_pdf(
        token=token,
        file_id=file_id
    )


@psb_bp.get("/getMyChosenTypeNodes")
def get_my_chosen_type_nodes_api():
    token = Req.receive_header_token()

    node_label = Req.receive_get_param("nodeLabel")

    return PSBService.get_my_chosen_type_nodes(
        token=token,
        node_label=node_label
    )


@psb_bp.get("/getMyAllNodes")
def get_my_all_nodes_api():
    token = Req.receive_header_token()

    return PSBService.get_my_all_nodes(
        token=token
    )


@psb_bp.get("/getMyFiles")
def get_my_files_api():
    token = Req.receive_header_token()

    return PSBService.get_my_files(
        token=token
    )


@psb_bp.post("/uploadFile")
def upload_file_api():
    token = Req.receive_header_token()

    file = Req.receive_file_param("file")
    file_name = Req.receive_form_param("file_name")
    file_size = Req.receive_form_param("file_size")
    file_type = Req.receive_form_param("file_type")

    return PSBService.upload_file(
        token=token,
        file=file,
        file_name=file_name,
        file_size=file_size,
        file_type=file_type
    )


@psb_bp.post("/addNode")
def add_node_api():
    token = Req.receive_header_token()

    chosen_node_label = Req.receive_post_param("chosenNodeLabel")
    chosen_node_id = Req.receive_post_param("chosenNodeId")
    related_node_label = Req.receive_post_param("relatedNodeLabel")
    related_text_name = Req.receive_post_param("relatedTextName")

    return PSBService.add_node(
        token=token,
        chosen_node_label=chosen_node_label,
        chosen_node_id=chosen_node_id,
        related_node_label=related_node_label,
        related_text_name=related_text_name
    )


