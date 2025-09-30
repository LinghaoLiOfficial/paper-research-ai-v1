import os

from buffer.GlobalInit import GlobalInit
from buffer.KnowledgeGraphTaskBuffer import KnowledgeGraphTaskBuffer
from mapper.LoginMapper import LoginMapper
from buffer.ScienceResearchProgress import ScienceResearchProgress
from utils.common.EmailSender import EmailSender
from utils.common.HashParser import HashParser
from utils.common.RandomStrGenerator import RandomStrGenerator
from utils.common.TimeParser import TimeParser
from utils.common.JWTParser import JWTParser
from entity.common.Resp import Resp


class LoginService:

    @classmethod
    def get_user_info(cls, token):

        if token is None:
            username = "默认"
            user_auth = "guest"
        else:
            # 解析token
            jwt_parser_result = JWTParser.decode_user_id(
                token=token
            )
            if not jwt_parser_result.check:
                return Resp.build_jwt_error(jwt_parser_result)

            user_id = jwt_parser_result.get_data_on_results()

            mysql_result = LoginMapper.select_user_where_user_id({
                "user_id": user_id
            })

            if len(mysql_result.get_data_on_results()) == 0:
                username = "默认"
                user_auth = "guest"
            else:
                user_data = mysql_result.get_data_on_results()[0]
                username = user_data['username']
                user_auth = user_data['user_auth']

        user_auth_mapping = {
            "guest": "游客",
            "user": "普通用户",
            "admin": "管理员"
        }

        user_auth_label = user_auth_mapping[user_auth]

        return Resp.build_success(data={
            "username": username,
            "userAuth": user_auth_label
        })

    @classmethod
    def cancel_all_tasks(cls, token):

        # 解析token
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        mysql_result = LoginMapper.select_user_where_user_id({
            "user_id": user_id
        })

        if mysql_result.get_data_on_results()[0]['user_auth'] != 'admin':
            return Resp.build_error(
                code=50001,
                message="您的权限等级不够，无法操作"
            )

        for common_user_id, user_data in ScienceResearchProgress.get_all().items():
            if user_data['total_progress']['now'] < user_data['total_progress']['all']:

                mysql_result1 = LoginMapper.select_user_where_user_id({
                    "user_id": common_user_id
                })

                user_email = mysql_result1.get_data_on_results()[0]['user_email']

                try:
                    # 发送任务强制终止信息到用户邮箱
                    EmailSender.send_knowledge_graph_task_cancel(
                        email=user_email,
                        task_id=user_data['current_task_id'],
                        task_name=user_data['current_task_name']
                    )
                except Exception as e:
                    print(e)
                    continue

        try:
            KnowledgeGraphTaskBuffer.clear_and_reinit()
        except Exception as e:
            print(e)
            return Resp.build_error()

        return Resp.build_success()

    @classmethod
    def get_all_knowledge_graph_task(cls, token):

        # 解析token
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        mysql_result = LoginMapper.select_user_where_user_id({
            "user_id": user_id
        })

        if mysql_result.get_data_on_results()[0]['user_auth'] != 'admin':
            return Resp.build_error(
                code=50001,
                message="您的权限等级不够，无法操作"
            )

        tasks_data = ScienceResearchProgress.get_all()

        tasks_headers = [
            "用户id",
            "用户名",
            "当前任务状态",
            "当前任务总进度",
            "当前任务阶段",
            "当前任务id",
            "当前任务名称",
            "日志信息"
        ]
        tasks = []
        for user_id, user_data in tasks_data.items():
            mysql_result = LoginMapper.select_user_where_user_id({
                "user_id": user_id
            })

            username = mysql_result.get_data_on_results()[0]['username']

            task_status = 'unknown'
            if user_data['current_task_id'] != "":
                mysql_result1 = LoginMapper.select_paper_search_where_user_id_and_task_id({
                    "user_id": user_id,
                    "task_id": user_data['current_task_id']
                })
                task_status = mysql_result1.get_data_on_results()[0]['task_status']

            # 日志
            log_list = [f"{log['timestamp']}: {log['message']}" for log in user_data['log']]

            tasks.append([
                user_id,
                username,
                task_status,
                f"{user_data['total_progress']['now']} / {user_data['total_progress']['all']}",
                user_data['current_stage'],
                user_data['current_task_id'],
                user_data['current_task_name'],
                # "\n".join([f"{log['timestamp']}: {log['message']}" for log in user_data['log']])
                log_list[0] if len(log_list) != 0 else ""
            ])

        return Resp.build_success(data={
            "knowledgeGraphTasks": tasks,
            "knowledgeGraphTasksHeaders": tasks_headers
        })

    @classmethod
    def add_new_activation(cls, token):

        # 解析token
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        mysql_result = LoginMapper.select_user_where_user_id({
            "user_id": user_id
        })

        if mysql_result.get_data_on_results()[0]['user_auth'] != 'admin':
            return Resp.build_error(
                code=50001,
                message="您的权限等级不够，无法操作"
            )

        add_num = 50

        for _ in range(add_num):

            code_id = RandomStrGenerator.generate_uuid()
            activation_code = RandomStrGenerator.generate_5_random_str()

            mysql_result1 = LoginMapper.insert_activation({
                "code_id": code_id,
                "activation_code": activation_code
            })

        return Resp.build_success()

    @classmethod
    def get_all_activation_code(cls, token):

        # 解析token
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        mysql_result = LoginMapper.select_user_where_user_id({
            "user_id": user_id
        })

        if mysql_result.get_data_on_results()[0]['user_auth'] != 'admin':
            return Resp.build_error(
                code=50001,
                message="您的权限等级不够，无法查看"
            )

        mysql_result1 = LoginMapper.select_all_activation({})

        codes = [[x for x in record.values()] for record in mysql_result1.get_data_on_results()]
        codes_headers = ["id", "激活码"]

        return Resp.build_success(data={
            "activationCodes": codes,
            "activationCodesHeaders": codes_headers
        })

    @classmethod
    def check_admin_token(cls, token):

        # 解析token
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        mysql_result = LoginMapper.select_user_where_user_id({
            "user_id": user_id
        })

        if mysql_result.get_data_on_results()[0]['user_auth'] != 'admin':
            return Resp.build_error(
                code=50001,
                message="您的权限等级不够，无法查看"
            )

        return Resp.build_success()

    @classmethod
    def check_token(cls, token):

        # 全局初始化任务
        if not GlobalInit.check():
            mysql_result = LoginMapper.reset_paper_search({})
            GlobalInit.update()

        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        return Resp.build_success()

    @classmethod
    def get_salt(cls, username):

        # 如果数据库中已经有盐值，则返回；否则生成一个盐值并返回

        mysql_result = LoginMapper.select_all_from_user_where_username({
            "username": username
        })
        if not mysql_result.check:
            return Resp.build_db_error()

        if not mysql_result.verify_data_on_results():
            salt = HashParser.generate_salt()

            return Resp.build_success(
                data={
                    "salt": salt
                }
            )

        salt = mysql_result.get_data_on_results()[0]['user_salt']

        return Resp.build_success(
            data={
                "salt": salt
            }
        )

    @classmethod
    def login(cls, username, password):

        # 根据用户名，查询用户
        mysql_result = LoginMapper.select_all_from_user_where_username({
            "username": username
        })
        if not mysql_result.check:
            return Resp.build_db_error()

        # 判断用户名是否存在
        if not mysql_result.verify_data_on_results():
            return Resp.build_error(
                code=50001,
                message="用户名不存在"
            )

        # 根据用户名+密码，查询用户
        mysql_result1 = LoginMapper.select_all_from_user_where_username_password({
            "username": username,
            "password": password
        })
        if not mysql_result1.check:
            return Resp.build_db_error()

        # 判断密码是否正确
        if not mysql_result1.verify_data_on_results():
            return Resp.build_error(
                code=50002,
                message="密码错误"
            )

        # 根据用户id+用户名，生成token
        jwt_parser_result = JWTParser.encode(
            user_id=mysql_result1.get_data_on_results()[0]['user_id'],
            username=mysql_result1.get_data_on_results()[0]['username']
        )
        if not jwt_parser_result.check:
            return Resp.build_error()

        token = jwt_parser_result.get_data_on_results()

        return Resp.build_success(
            data={
                "token": token
            }
        )

    @classmethod
    def send_email(cls, email):

        # 根据邮箱，查询邮箱是否已被注册
        mysql_result = LoginMapper.select_all_from_user_where_email({
            "email": email
        })
        if not mysql_result.check:
            return Resp.build_db_error()

        if mysql_result.verify_data_on_results():
            return Resp.build_error(
                code=50001,
                message="该邮箱已被注册"
            )

        # 查询邮箱发送是否过于频繁
        mysql_result1 = LoginMapper.select_all_from_validation_where_email({
            "email": email
        })
        if not mysql_result1.check:
            return Resp.build_db_error()

        if mysql_result1.verify_data_on_results():

            time = mysql_result1.get_data_on_results()[0]['validation_create_timestamp']

            time_calculator_result = TimeParser.calculate_passing_time(time)

            if not time_calculator_result.check:
                return Resp.build_error(
                    code=50002,
                    message=f"邮箱发送过于频繁，请于{os.getenv('VALIDATION_REPEAT_SENDING_TIME_INTERVAL')}分钟之后重新发送"
                )

        # 事务删除旧的邮箱验证码
        mysql_result2 = LoginMapper.transaction_delete_validation({
            "validation_email": email
        })
        if not mysql_result2.check:
            return Resp.build_db_error()

        # 生成新的邮箱验证码
        code = RandomStrGenerator.generate_validation_code()

        # 事务插入新的邮箱验证码
        mysql_result3 = LoginMapper.transaction_insert_validation(
            {
                "id": RandomStrGenerator.generate_uuid(),
                "email": email,
                "code": code,
            }
        )
        if not mysql_result3.check:
            return Resp.build_db_error()

        # 发送邮箱
        email_sender_result = EmailSender.send_validation_code(email, code)

        if not email_sender_result.check:

            return Resp.build_error(
                code=50003,
                message="邮箱发送失败，请重试"
            )

        return Resp.build_success()

    @classmethod
    def register(cls, username, password, email, code, salt, activation_code):

        # 根据邮箱，判断邮箱验证码是否存在
        mysql_result = LoginMapper.select_all_from_validation_where_email({
            "email": email
        })
        if not mysql_result.check:
            return Resp.build_db_error()

        # 根据邮箱和验证码，判断邮箱验证码是否正确
        if not mysql_result.verify_data_on_results():
            return Resp.build_error(
                code=50001,
                message="请先点击发送邮箱"
            )

        # 事务删除邮箱验证码信息
        if code != mysql_result.get_data_on_results()[0]['validation_code']:
            return Resp.build_error(
                code=50002,
                message="邮箱验证码有误"
            )

        # 激活码检查
        mysql_result0 = LoginMapper.select_activation({
            "activation_code": activation_code
        })

        if len(mysql_result0.get_data_on_results()) == 0:
            return Resp.build_error(
                code=50003,
                message="激活码无效"
            )

        mysql_result1 = LoginMapper.transaction_delete_validation({
            "validation_email": email
        })
        if not mysql_result1.check:
            return Resp.build_db_error()

        # 生成用户ID
        user_id = RandomStrGenerator.generate_uuid()

        # 事务插入用户信息
        mysql_result2 = LoginMapper.transaction_insert_user({
            "user_id": user_id,
            "username": username,
            "password": password,
            "auth": "user",
            "email": email,
            "salt": salt,
            "user_activation": activation_code
        })
        if not mysql_result2.check:
            return Resp.build_db_error()

        # 事务生成新的用户节点
        neo4j_result = LoginMapper.transaction_merge_user_node({
            "user_id": user_id,
            "username": username,
        })
        if not neo4j_result.check:
            return Resp.build_db_error()

        # 激活码删除
        mysql_result3 = LoginMapper.delete_activation({
            "activation_code": activation_code
        })

        # 创建保存用户文件用的文件夹
        os.makedirs(f"./storage/{user_id}", exist_ok=True)
        user_zone_list = [
            "data_analysis",
            "knowledge_base",
            "science_research",
            "web_crawl"
        ]
        for zone in user_zone_list:
            os.makedirs(f"./storage/{user_id}/{zone}", exist_ok=True)

        return Resp.build_success()


