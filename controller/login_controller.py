from flask import Blueprint

from service.LoginService import LoginService
from entity.common.Req import Req

# 实例化Blueprint
login_bp = Blueprint(
    name="login",
    import_name=__name__,
    url_prefix="/login"
)


# 一键终止所有任务
@login_bp.post("/cancelAllTasks")
def cancel_all_tasks_api():
    token = Req.receive_header_token()

    return LoginService.cancel_all_tasks(
        token=token
    )


# 获取排队中的论文知识网络构建任务
@login_bp.get("/getAllKnowledgeGraphTask")
def get_all_knowledge_graph_task_api():
    token = Req.receive_header_token()

    return LoginService.get_all_knowledge_graph_task(
        token=token
    )


# 添加激活码
@login_bp.post("/addNewActivationCode")
def add_new_activation_code_api():
    token = Req.receive_header_token()

    return LoginService.add_new_activation(
        token=token
    )


# 获取激活码
@login_bp.get("/getAllActivationCode")
def get_all_activation_code_api():
    token = Req.receive_header_token()

    return LoginService.get_all_activation_code(
        token=token
    )


# 获取用户信息
@login_bp.get("/getUserInfo")
def get_user_info_api():
    token = Req.receive_header_token()

    return LoginService.get_user_info(
        token=token
    )


# 检查管理员Token
@login_bp.get("/checkAdminToken")
def check_admin_token_api():
    token = Req.receive_header_token()

    return LoginService.check_admin_token(
        token=token
    )


# 检查Token
@login_bp.get("/checkToken")
def check_token_api():
    token = Req.receive_header_token()

    return LoginService.check_token(
        token=token
    )


# 获取盐值
@login_bp.get("/getSalt")
def get_salt_api():
    username = Req.receive_get_param("username")

    return LoginService.get_salt(
        username=username
    )


# 登录
@login_bp.post("/login")
def login_api():
    username = Req.receive_post_param("username")
    password = Req.receive_post_param("password")

    return LoginService.login(
        username=username,
        password=password
    )


# 发送邮件
@login_bp.post("/sendEmail")
def send_email_api():
    email = Req.receive_post_param("email")

    return LoginService.send_email(
        email=email
    )


# 注册
@login_bp.post("/register")
def register_api():
    username = Req.receive_post_param("username")
    password = Req.receive_post_param("password")
    email = Req.receive_post_param("email")
    code = Req.receive_post_param("code")
    salt = Req.receive_post_param("salt")
    activation_code = Req.receive_post_param("activationCode")

    return LoginService.register(
        username=username,
        password=password,
        email=email,
        code=code,
        salt=salt,
        activation_code=activation_code
    )






