from flask import Blueprint

from service.LoginService import LoginService
from entity.common.Req import Req

sda_bp = Blueprint(
    name="smart_data_analysis",
    import_name=__name__,
    url_prefix="/sda"
)


@sda_bp.get("/getSelectDataset")
def get_select_dataset_api():
    username = Req.receive_get_param("username")

    return LoginService.get_salt(
        username=username
    )
